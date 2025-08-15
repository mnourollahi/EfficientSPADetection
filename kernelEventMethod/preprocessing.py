from collections import deque, defaultdict
import bt2
import pandas as pd
from pathlib import Path
import sys

def iterate_kerneltraces_to_df(filepath):
    print(f"Iterating kernel traces from {filepath}...")

    df_columns = ['name', 'cur_ts', 'ret', 'tid', 'pid', 'vtid', 'vpid']
    data = []

    msg_it = bt2.TraceCollectionMessageIterator(filepath)

    for msg in msg_it:
        if isinstance(msg, bt2._EventMessageConst):
            event = msg.event
            cur_ts = msg.default_clock_snapshot.ns_from_origin
            ret_value = None  # Initialize ret_value as None

            # Check if the 'ret' field exists in the event payload
            if 'ret' in event.payload_field:
                ret_value = event.payload_field['ret']
            data.append(
                [event.name, cur_ts, ret_value, event['tid'], event['pid'], event['vtid'],
                 event['vpid']]
            )

    if not data:
        print(f"No data found in kernel traces from {filepath}.")
        return pd.DataFrame(columns=df_columns)

    df_kernel = pd.DataFrame(data, columns=df_columns)
    df = build_call_stack_kernel(df_kernel)
    print(f"Kernel traces converted to DataFrame for {filepath}.")

    return df

def build_call_stack_kernel(df):
    df['entry_ts'] = None
    df['exit_ts'] = None
    df['response_time'] = None

    syscall_events = {
        'syscall_entry_recvfrom': 'syscall_exit_recvfrom',
        'syscall_entry_recvmsg': 'syscall_exit_recvmsg',
        'syscall_entry_recvmmsg': 'syscall_exit_recvmmsg',
        'syscall_entry_sendto': 'syscall_exit_sendto',
        'syscall_entry_sendmsg': 'syscall_exit_sendmsg',
        'syscall_entry_sendmmsg': 'syscall_exit_sendmmsg',
    }

    stack = defaultdict(deque)
    merged_data = []

    for row in df.itertuples(index=False):
        event_name = row.name
        vtid = row.vtid

        if event_name in syscall_events:
            stack[(vtid, event_name)].append(row)
        elif event_name in syscall_events.values():
            for entry_event, entry_rows in list(stack.items()):
                if entry_event[0] == vtid and syscall_events[entry_event[1]] == event_name:
                    entry_row = entry_rows.pop()
                    response_time = row.cur_ts - entry_row.cur_ts
                    merged_row = entry_row._asdict()
                    merged_row.update({
                        'entry_ts': entry_row.cur_ts,
                        'exit_ts': row.cur_ts,
                        'response_time': response_time,
                        'ret': row.ret,
                    })
                    merged_data.append(merged_row)
                    if not entry_rows:
                        del stack[entry_event]
                    break

    df_merged = pd.DataFrame(merged_data)
    if not df_merged.empty:
        df_merged.sort_values("entry_ts", inplace=True)
        df_merged.reset_index(drop=True, inplace=True)

    return df_merged

def iterate_traces_to_df(filepath, first_ts, last_ts):
    df_columns = ['e_name', 'cur_ts', 'start_time', 'span_id', 'op_name', 'service_name', 'parent_span_id', 'duration']
    data = []

    start_spans = {}
    msg_it = bt2.TraceCollectionMessageIterator(filepath)
    for msg in msg_it:
        if isinstance(msg, bt2._EventMessageConst):
            cur_ts = msg.default_clock_snapshot.ns_from_origin
            if first_ts <= cur_ts <= last_ts:
                event = msg.event
                if event.name == 'jaeger_ust:start_span':
                    start_spans[event['span_id']] = [
                        event.name, cur_ts, event['start_time'], event['span_id'], event['op_name'], event['service_name'],
                        event['parent_span_id'], None
                    ]
                elif event.name == 'jaeger_ust:end_span':
                    span_id = event['span_id']
                    if span_id in start_spans:
                        start_row = start_spans.pop(span_id)
                        start_row[-1] = event['duration']  # Update duration from end_span event
                        data.append(start_row)

    if not data:
        print(f"No data found in UST traces from {filepath}.")
        return pd.DataFrame(columns=df_columns)

    df = pd.DataFrame(data, columns=df_columns)
    return df

def merge_dfs(df_enhanced, df_ust):
    df_ust_expanded = df_ust.assign(
        name=df_ust['e_name'],
        response_time=df_ust['duration'],
        ret= 0,
    )
    df_ust_expanded = df_ust_expanded[['name', 'cur_ts', 'ret' , 'response_time', 'span_id', 'parent_span_id', 'op_name', 'service_name']]

    df_enhanced_expanded = df_enhanced.assign(
        span_id="Null",
        parent_span_id="Null",
        op_name=df_enhanced['name'],
        service_name= "Null")
    df_enhanced_expanded = df_enhanced_expanded[['name', 'cur_ts', 'ret', 'response_time', 'span_id', 'parent_span_id', 'op_name', 'service_name']]


    df_merged = pd.concat([df_ust_expanded, df_enhanced_expanded], ignore_index=True).sort_values('cur_ts').reset_index(drop=True)
    df_merged['ret'] = df_merged['ret'].apply(lambda x: 0 if pd.isna(x) or x < 0 else x)
    return df_merged

    def process_merged_events(df_merged, load_number):
        read_user_timeline_spans = {}
        event_details = []

        # First pass: Identify ReadUserTimeline spans and their corresponding time intervals
        for row in df_merged.itertuples(index=True):
            if row.op_name == 'ReadUserTimeline':
                span_id = row.span_id
                start_time = int(row.cur_ts)
                end_time = start_time + int(row.response_time)
                read_user_timeline_spans[span_id] = {
                    'index': row.Index,  # Use row.Index to access the index in itertuples
                    'start_time': start_time,
                    'response_time': row.response_time,
                    'end_time': end_time,
                    'details': row._asdict()  # Faster than to_dict() when iterating with itertuples
                }

        # Second pass: Collect events within the identified ReadUserTimeline spans
        for row in df_merged.itertuples(index=False):
            event_time = int(row.cur_ts)
            for span_id, span_info in read_user_timeline_spans.items():
                if span_info['start_time'] <= event_time <= span_info['end_time']:
                    event_info = row._asdict()
                    if row.name != 'jaeger_ust:start_span':
                        event_info['parent_span_id'] = span_id
                    if 'events' not in span_info:
                        span_info['events'] = []
                    span_info['events'].append(event_info)

        # Construct event details for the final DataFrame
        for span_id, span_info in read_user_timeline_spans.items():
            if 'events' in span_info:
                event_details.append({
                    'load': load_number,
                    'op_name': span_info['details']['op_name'],
                    'service_name': span_info['details']['service_name'],
                    'event_op_names': [event['op_name'] for event in span_info['events']],
                    'event_service_names': [event['service_name'] for event in span_info['events']],
                    'event_response_times': [event['response_time'] for event in span_info['events']],
                    'event_rets': [event['ret'] for event in span_info['events']]
                })

        event_details_df = pd.DataFrame(event_details)

        # Extract unique op_name and service_name values and assign unique numbers
        unique_op_names = pd.unique(event_details_df['op_name'].explode())
        event_op_name_mapping = {op_name: idx for idx, op_name in enumerate(unique_op_names, start=1)}

        unique_service_names = pd.unique(event_details_df['service_name'].explode())
        service_name_mapping = {service_name: idx for idx, service_name in enumerate(unique_service_names, start=1)}

        # Function to map each element in the list based on the dynamically created mappings
        def map_event_names(event_list):
            return [event_op_name_mapping.get(event, 0) for event in event_list]

        def map_service_names(service_list):
            return [service_name_mapping.get(service, 0) for service in service_list]

        event_details_df['event_op_names_ids'] = event_details_df['event_op_names'].apply(map_event_names)
        event_details_df['event_service_names_ids'] = event_details_df['event_service_names'].apply(map_service_names)

        # Optionally, you can print or save the mappings for verification purposes
        print("Mapping of op_names to unique IDs:")
        print(event_op_name_mapping)

        print("Mapping of service_names to unique IDs:")
        print(service_name_mapping)

        # Remove unnecessary columns
        columns_to_drop = ['span_id', 'op_name_id', 'event_cur_ts_id', 'event_cur_ts', 'cur_ts', 'op_name',
                           'event_op_names', 'service_name', 'event_service_names']
        df = event_details_df.drop(columns=columns_to_drop, errors='ignore')

        # Save the final dataframe to a CSV file
        df.to_csv("processed_events.txt", encoding='utf-8', index=False)
        return df

def process_subfolder(kernel_path, ust_path, load_number):
    df_sorted_kernel = iterate_kerneltraces_to_df(str(kernel_path))

    if df_sorted_kernel.empty:
        return None, None

    first_ts = df_sorted_kernel["cur_ts"].min()
    last_ts = df_sorted_kernel["cur_ts"].max()

    df_ust = iterate_traces_to_df(str(ust_path), first_ts, last_ts)

    if df_ust.empty:
        return None, None

    df_merged = merge_dfs(df_sorted_kernel, df_ust)
    processed_events_df = process_merged_events(df_merged, load_number)

    return df_merged, processed_events_df

def process_folders(root_dir):
    root_path = Path(root_dir)

    for folder_path in root_path.iterdir():
        if folder_path.is_dir():
            folder_number = folder_path.name.replace('folder_', '')

            folder_df_merged = []
            folder_processed_events_df = []

            for subfolder_path in folder_path.iterdir():
                if subfolder_path.is_dir():
                    kernel_path = subfolder_path / 'kernel'
                    ust_path = subfolder_path / 'ust'

                    if kernel_path.exists() and ust_path.exists():
                        df_merged, processed_events_df = process_subfolder(kernel_path, ust_path, folder_number)

                        if df_merged is not None:
                            folder_df_merged.append(df_merged)
                        if processed_events_df is not None:
                            folder_processed_events_df.append(processed_events_df)

            if folder_df_merged:
                final_df_merged = pd.concat(folder_df_merged, ignore_index=True)
                final_df_merged.to_csv(f'final_df_merged_{folder_number}.txt', index=False)

            if folder_processed_events_df:
                final_processed_events_df = pd.concat(folder_processed_events_df, ignore_index=True)
                final_processed_events_df.to_csv(f'final_processed_events_df_{folder_number}.txt', index=False)

# Main execution point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <root_directory>")
        sys.exit(1)

    root_directory = sys.argv[1]
    process_folders(root_directory)
