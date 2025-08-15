from collections import defaultdict, deque
import bt2
import pandas as pd
from pathlib import Path
import sys

def iterate_kerneltraces_to_df(filepath):
    print(f"Iterating kernel traces from {filepath}...")
    filepath = str(filepath)
    df_columns = ['name', 'cur_ts', 'ret', 'tid', 'pid', 'vtid', 'vpid', 'trace_id', 'service_name']
    data = []

    msg_it = bt2.TraceCollectionMessageIterator(filepath)
    for msg in msg_it:
        if isinstance(msg, bt2._EventMessageConst):
            event = msg.event
            cur_ts = msg.default_clock_snapshot.ns_from_origin
            ret_value = event.payload_field.get('ret', None)
            data.append([event.name, cur_ts, ret_value, event['tid'], event['pid'], event['vtid'], event['vpid'], None, None])

    return pd.DataFrame(data, columns=df_columns) if data else pd.DataFrame(columns=df_columns)

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
            entry_key = (vtid, next(k for k in syscall_events if syscall_events[k] == event_name))
            if stack[entry_key]:
                entry_row = stack[entry_key].pop()
                merged_row = entry_row._asdict()
                merged_row.update({
                    'entry_ts': entry_row.cur_ts,
                    'exit_ts': row.cur_ts,
                    'response_time': row.cur_ts - entry_row.cur_ts,
                    'ret': row.ret,
                })
                merged_data.append(merged_row)
                if not stack[entry_key]:
                    del stack[entry_key]

    return pd.DataFrame(merged_data) if merged_data else pd.DataFrame(columns=df.columns)

def iterate_traces_to_df(filepath):
    filepath = str(filepath)
    df_columns = ['e_name', 'cur_ts', 'start_time', 'span_id', 'op_name', 'service_name', 'parent_span_id', 'duration', 'trace_id', 'vtid']
    data, start_spans = [], {}

    msg_it = bt2.TraceCollectionMessageIterator(filepath)
    for msg in msg_it:
        if isinstance(msg, bt2._EventMessageConst):
            cur_ts = msg.default_clock_snapshot.ns_from_origin
            event = msg.event
            trace_id = event['trace_id_low']
            vtid = event['vtid']
            span_id = event['span_id']

            if event.name == 'jaeger_ust:start_span':
                start_spans[span_id] = [event.name, cur_ts, event['start_time'], span_id, event['op_name'], event['service_name'], event['parent_span_id'], None, trace_id, vtid]
            elif event.name == 'jaeger_ust:end_span' and span_id in start_spans:
                start_row = start_spans.pop(span_id)
                start_row[-3] = event['duration']
                data.append(start_row)

    return pd.DataFrame(data, columns=df_columns)

def merge_dfs(df_kernel, df_ust):
    # Group UST events by trace_id to get start and end timestamps of each trace
    ust_groups = df_ust.groupby('trace_id').agg(
        start_ts=('cur_ts', 'min'),
        end_ts=('cur_ts', 'max'),
        service_name=('service_name', 'first')
    ).reset_index()
    df_kernel['span_id'] = None

    # Merge UST group info back to the original UST DataFrame to retain vtid info
    df_ust = pd.merge(df_ust, ust_groups[['trace_id', 'start_ts', 'end_ts']], on='trace_id', how='left')

    df_ust['entry_ts'] = df_ust['cur_ts']
    df_ust['exit_ts'] = df_ust['cur_ts'] + df_ust['response_time']


    # Apply trace_id and service_name to kernel events within the timestamp range and matching vtid
    for _, ust_group in ust_groups.iterrows():
        trace_id = ust_group['trace_id']
        start_ts = ust_group['start_ts']
        end_ts = ust_group['end_ts']
        service_name = str(ust_group['service_name'])  # Explicitly cast service_name to a scalar string

        # Filter kernel events within the start and end timestamp range and matching vtid
        mask = (df_kernel['cur_ts'] >= start_ts) & (df_kernel['cur_ts'] <= end_ts) & \
               (df_kernel['vtid'].isin(df_ust[df_ust['trace_id'] == trace_id]['vtid']))

        # Assign trace_id and service_name directly without list replication
        df_kernel.loc[mask, 'trace_id'] = trace_id
        df_kernel.loc[mask, 'service_name'] = service_name  # Explicitly a scalar

    # Filter out kernel events that didn't get a trace_id assigned
    df_kernel = df_kernel[df_kernel['trace_id'].notna()]
    for i, k_event in df_kernel.iterrows():
        matching_ust = df_ust[(df_ust['entry_ts'] <= k_event['cur_ts']) & (df_ust['exit_ts'] >= k_event['cur_ts'])]
        if not matching_ust.empty:
            df_kernel.at[i, 'span_id'] = matching_ust['span_id'].values[0]  # Assigning first matching span_id

    # Prepare DataFrames for concatenation
    df_kernel_filtered = df_kernel[['name', 'cur_ts', 'ret', 'response_time', 'span_id', 'op_name', 'service_name', 'trace_id']]
    df_ust_expanded = df_ust.rename(columns={'e_name': 'name', 'duration': 'response_time'})
    df_ust_expanded['ret'] = 0
    df_ust_expanded = df_ust_expanded[['name', 'cur_ts', 'ret', 'response_time', 'span_id', 'op_name', 'service_name', 'trace_id']]

    df_merged = pd.concat([df_ust_expanded, df_kernel_filtered], ignore_index=True).sort_values('cur_ts').reset_index(drop=True)
    df_merged['ret'] = df_merged['ret'].fillna(0).clip(lower=0)
    return df_merged


def process_subfolder(kernel_path, ust_path, load_number):
    df_kernel = iterate_kerneltraces_to_df(kernel_path)
    print("processing kernel")
    if df_kernel.empty:
        return None

    df_ust = iterate_traces_to_df(ust_path)
    if df_ust.empty:
        return None

    return merge_dfs(df_kernel, df_ust)

def process_folders(root_dir):
    root_path = Path(root_dir)
    print("processing root")

    for folder_path in root_path.iterdir():
        if folder_path.is_dir():
            folder_number = folder_path.name.replace('folder_', '')

            folder_df_merged = []
            for subfolder_path in folder_path.iterdir():
                if subfolder_path.is_dir():
                    kernel_path, ust_path = subfolder_path / 'kernel', subfolder_path / 'ust'
                    if kernel_path.exists() and ust_path.exists():
                        df_merged = process_subfolder(kernel_path, ust_path, folder_number)
                        if df_merged is not None:
                            folder_df_merged.append(df_merged)

            if folder_df_merged:
                pd.concat(folder_df_merged, ignore_index=True).to_csv(f'final_df_merged_{folder_number}.txt', index=False)

# Main execution point
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python script.py <root_directory>")
        sys.exit(1)

    root_directory = sys.argv[1]
    process_folders(root_directory)
