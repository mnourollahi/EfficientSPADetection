# /*I want to build a merged dataset of ust and kernel events with these columns:
# df_enhanced_expanded = df_enhanced_expanded[['name', 'cur_ts', 'ret', 'response_time', 'span_id', 'parent_span_id', 'op_name', 'service_name', 'trace_id']]
#
# I have kernel events in this format:
# ['name', 'cur_ts', 'ret' , 'response_time', 'span_id', 'parent_span_id', 'op_name']
# and ust events in this columns:
# ['name', 'cur_ts', 'response_time', 'span_id', 'parent_span_id', 'op_name', 'service_name', 'trace_id']
#
# as you see the kernel events donèt have  'service_name', 'trace_id which is required in the final results
# a group of ust events that have the same trace_id form one trace
# each trace has a start time and end time which is the same as start time of the first event in the trace, and end time of the last event
# any kernel events that happen in between start and end time of a ust trace is related to that trace, so their trace_id should be the same the ust trace trace_id, if vtid of the kernel event is the same as one of the vtid of the events in that ust trace
# in addition, service_name of the kernel event is the same as the ust event which its trace_id and vtid is the same as the kernel event*/

from collections import deque, defaultdict
import bt2
import pandas as pd
from pathlib import Path
import sys

def iterate_kerneltraces_to_df(filepath):
    print(f"Iterating kernel traces from {filepath}...")

    # Ensure filepath is a string, not PosixPath
    filepath = str(filepath)

    df_columns = ['name', 'cur_ts', 'ret', 'tid', 'pid', 'vtid', 'vpid', 'trace_id', 'service_name']
    data = []

    msg_it = bt2.TraceCollectionMessageIterator(filepath)

    for msg in msg_it:
        if isinstance(msg, bt2._EventMessageConst):
            event = msg.event
            cur_ts = msg.default_clock_snapshot.ns_from_origin
            ret_value = event.payload_field.get('ret', None)  # Use get() to avoid key errors
            trace_id = None
            service_name = None

            data.append(
                [event.name, cur_ts, ret_value, event['tid'], event['pid'], event['vtid'], event['vpid'], trace_id, service_name]
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
            entry_event = (vtid, next(k for k in syscall_events if syscall_events[k] == event_name))
            if stack[entry_event]:
                entry_row = stack[entry_event].pop()
                response_time = row.cur_ts - entry_row.cur_ts
                merged_row = entry_row._asdict()
                merged_row.update({
                    'entry_ts': entry_row.cur_ts,
                    'exit_ts': row.cur_ts,
                    'response_time': response_time,
                    'ret': row.ret,
                })
                merged_data.append(merged_row)

                if not stack[entry_event]:
                    del stack[entry_event]

    df_merged = pd.DataFrame(merged_data)
    if not df_merged.empty:
        df_merged.sort_values("entry_ts", inplace=True)
        df_merged.reset_index(drop=True, inplace=True)

    return df_merged


def iterate_traces_to_df(filepath):
    # Convert PosixPath to string
    filepath = str(filepath)

    df_columns = ['e_name', 'cur_ts', 'start_time', 'span_id', 'op_name', 'service_name', 'parent_span_id', 'duration',
                  'trace_id', 'vtid']
    data = []

    start_spans = {}
    msg_it = bt2.TraceCollectionMessageIterator(filepath)
    for msg in msg_it:
        if isinstance(msg, bt2._EventMessageConst):
            cur_ts = msg.default_clock_snapshot.ns_from_origin
            event = msg.event
            trace_id = event['trace_id_low']
            vtid = event['vtid']

            # Check for the start of a span
            if event.name == 'jaeger_ust:start_span':
                start_spans[event['span_id']] = [
                    event.name, cur_ts, event['start_time'], event['span_id'], event['op_name'],
                    event['service_name'], event['parent_span_id'], None, trace_id, vtid
                ]

            # Check for the end of a span
            elif event.name == 'jaeger_ust:end_span':
                span_id = event['span_id']
                if span_id in start_spans:
                    start_row = start_spans.pop(span_id)
                    start_row[-3] = event['duration']  # Update duration from end_span event
                    data.append(start_row)

    return pd.DataFrame(data, columns=df_columns)


def merge_dfs(df_kernel, df_ust):
    # Ensure that the kernel dataframe has the necessary columns
    required_columns = ['span_id', 'parent_span_id', 'op_name']
    for col in required_columns:
        if col not in df_kernel.columns:
            df_kernel[col] = None  # Initialize as None if the column is missing

    # Group UST events by trace_id to find the start and end timestamps of each trace
    ust_groups = df_ust.groupby('trace_id').agg(
        start_ts=('cur_ts', 'min'),  # Start timestamp of the trace
        end_ts=('cur_ts', 'max'),  # End timestamp of the trace
        service_name=('service_name', 'first'),  # Service name for this trace
    ).reset_index()

    # Merge the UST group info back to the original UST dataframe to retain the vtid info
    df_ust = pd.merge(df_ust, ust_groups[['trace_id', 'start_ts', 'end_ts']], on='trace_id', how='left')

    # Initialize columns for trace_id and service_name in the kernel DataFrame
    df_kernel['trace_id'] = None
    df_kernel['service_name'] = None

    # Dictionary to store UST data per trace_id for quick lookup
    ust_dict = defaultdict(list)

    # Iterate over UST events and store them in ust_dict by trace_id
    for _, ust_event in df_ust.iterrows():
        trace_id = ust_event['trace_id']
        ust_dict[trace_id].append(ust_event)

    # Now iterate over each trace_id group in ust_groups (faster than nested loops)
    for _, ust_group in ust_groups.iterrows():
        trace_id = ust_group['trace_id']
        start_ts = ust_group['start_ts']
        end_ts = ust_group['end_ts']
        service_name = str(ust_group['service_name'])  # Ensure service_name is a scalar string

        # Filter kernel events that fall within the start and end timestamp range
        mask = (df_kernel['cur_ts'] >= start_ts) & (df_kernel['cur_ts'] <= end_ts)
        kernel_filtered = df_kernel[mask]

        # Get the list of UST events for the current trace_id
        ust_events = ust_dict[trace_id]

        # For each kernel event, match by vtid
        for ust_event in ust_events:
            vtid = ust_event['vtid']

            # Find kernel events with matching vtid
            kernel_vtid_mask = (kernel_filtered['vtid'] == vtid)

            # Apply trace_id and service_name to matching kernel events
            df_kernel.loc[kernel_filtered.index[kernel_vtid_mask], 'trace_id'] = trace_id
            df_kernel.loc[kernel_filtered.index[kernel_vtid_mask], 'service_name'] = service_name

    # Filter out kernel events that didn't get a trace_id assigned
    df_kernel = df_kernel[df_kernel['trace_id'].notnull()]

    # Return the final merged dataframe with required columns
    df_enhanced = df_kernel[
        ['name', 'cur_ts', 'ret', 'response_time', 'span_id', 'parent_span_id', 'op_name', 'service_name', 'trace_id']]

    df_ust_expanded = df_ust.assign(
        name=df_ust['e_name'],
        response_time=df_ust['duration'],
        ret=0,
    )
    df_ust_expanded = df_ust_expanded[
        ['name', 'cur_ts', 'ret', 'response_time', 'span_id', 'parent_span_id', 'op_name', 'service_name', 'trace_id']]

    df_enhanced_expanded = df_enhanced.assign(
        span_id="Null",
        parent_span_id="Null",
        op_name=df_enhanced['name'])
    df_enhanced_expanded = df_enhanced_expanded[
        ['name', 'cur_ts', 'ret', 'response_time', 'span_id', 'parent_span_id', 'op_name', 'service_name', 'trace_id']]

    df_merged = pd.concat([df_ust_expanded, df_enhanced_expanded], ignore_index=True).sort_values(
        'cur_ts').reset_index(drop=True)
    df_merged['ret'] = df_merged['ret'].apply(lambda x: 0 if pd.isna(x) or x < 0 else x)
    return df_merged


def process_subfolder(kernel_path, ust_path, load_number):
    df_kernel = iterate_kerneltraces_to_df(kernel_path)
    if df_kernel.empty:
        return None

    df_ust = iterate_traces_to_df(ust_path)
    if df_ust.empty:
        return None

    df_merged = merge_dfs(df_kernel, df_ust)
    return df_merged


def process_folders(root_dir):
    root_path = Path(root_dir)

    for folder_path in root_path.iterdir():
        if folder_path.is_dir():
            folder_number = folder_path.name.replace('folder_', '')

            folder_df_merged = []

            for subfolder_path in folder_path.iterdir():
                if subfolder_path.is_dir():
                    kernel_path = subfolder_path / 'kernel'
                    ust_path = subfolder_path / 'ust'

                    if kernel_path.exists() and ust_path.exists():
                        df_merged = process_subfolder(kernel_path, ust_path, folder_number)

                        if df_merged is not None:
                            folder_df_merged.append(df_merged)

            if folder_df_merged:
                final_df_merged = pd.concat(folder_df_merged, ignore_index=True)
                final_df_merged.to_csv(f'final_df_merged_{folder_number}.txt', index=False)


# Main execution point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <root_directory>")
        sys.exit(1)

    root_directory = sys.argv[1]
    process_folders(root_directory)
