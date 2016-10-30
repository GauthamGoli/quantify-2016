# Our algorithm has a time complexity of logarithmic order
# But the memory complexity is high, due to which were running out of memory for last few large test cases, which could be avoided by using Square Root Decomposition Algorithm
# We got a score of 76.17 on hackerrank, although we learnt later that GS people had their own test cases in a different environment

import sys

# Global variable
# Assigns unique id to each job as key and value is the job itself, so each job is stored only once
# master_job_dict = {custom_job_id: [timestamp1,id1,orig1,instr,imp1,duration1], custom_job_id:[...]...}
master_job_dict = {}

# orig_list is unique, and origin's are reused
orig_list = []
# instr_list is unique, and instructions are reused
instr_list = []


def reverse_bisect_right(a, x, key, lo=0, hi=None):
    """Insert item x in list a, and keep it reverse-sorted assuming a
    is reverse-sorted.

    If x is already in a, insert it to the right of the rightmost x.

    Optional arg    s lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.

    Runtime: O(logn)
    """
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if x > master_job_dict[a[mid]][key]: hi = mid
        else: lo = mid+1
    return lo


def reverse_bisect_left(a, x, key, lo=0, hi=None):
    """Insert item x in list a, and keep it reverse-sorted assuming a
    is reverse-sorted.

    If x is already in a, insert it to the left of the leftmost x.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.

    Runtime: O(logn)
    """
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if master_job_dict[a[mid]][key] > x: lo = mid+1
        else: hi = mid
    return lo


def bisect_left(a, x, key, lo=0, hi=None):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(x) will
    insert just before the leftmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.

    Runtime: O(logn)
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if master_job_dict[a[mid]][key] < x: lo = mid+1
        else: hi = mid
    return lo


def bisect_right(a, x, key, lo=0, hi=None):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(x) will
    insert just after the rightmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.

    Runtime: O(logn)
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if x < master_job_dict[a[mid]][key]: hi = mid
        else: lo = mid+1
    return lo

def print_job(job_id):
    """
    prints the job by fetching values from data structures
    :param job_id:
    :return:
    """
    p = list(master_job_dict[job_id])
    p[2] = orig_list[p[2]]
    p[3] = instr_list[p[3]]
    print 'job ' + ' '.join(map(str, p[:6]))

lines = sys.stdin.readlines()

# Number of CPUs
num_cpu = int(lines[0].strip().split()[1])

jobs_queue = []        # Current state of queue: [ ,.....]
jobs_processing = []   # Jobs which are assigned to CPUs
jobs_history = {}      # State of queue at a timestamp:   jobs_history[time1] == state of jobs_queue at time1


new_job_id = 0         # custom job id

lines.pop(0)

for line in lines:

    parts = line.strip().split()
    if parts[0] == 'job':
        timestamp, id, orig, instr, imp, duration = int(parts[1]), int(parts[2]), parts[3],\
                                                    parts[4], int(parts[5]), int(parts[6])

        # Check if origin is already in orig_list, else append
        # Runtime: O(number of unique origins)
        if orig in orig_list:
            orig_id = orig_list.index(orig)
        else:
            orig_id = len(orig_list)
            orig_list.append(orig)

        # Check if instruction is already in instr_list, else append
        # Runtime: O(number of unique instructions)
        if instr in instr_list:
            instr_id = instr_list.index(instr)
        else:
            instr_id = len(instr_list)
            instr_list.append(instr)

        # Assign the job to master_job_dict
        new_job = [timestamp, id, orig_id, instr_id, imp, duration]
        master_job_dict[new_job_id] = new_job

        if len(jobs_queue):
            # Assuming that job_queue is already sorted, find the position to
            # insert new job in the job_queue using "BISECTION ALGORITHM"
            # In Worst case, bisection algorithm would be called 6 times
            # (2 times each for importance,timestamp,duration respectively)

            # Total Runtime: O(6logn) = O(logn)

            lo = bisect_left(jobs_queue, imp, 4)
            hi = bisect_right(jobs_queue, imp, 4)
            if lo == hi:
                # Importance is unique , Just insert in the right position
                jobs_queue.insert(lo, new_job_id)

            elif lo != hi:
                # There are already entries with same importance
                # Check for timestamp comparison

                # Runtime: O(logn)

                lo1 = reverse_bisect_left(jobs_queue, timestamp, 0, lo, hi)
                hi1 = reverse_bisect_right(jobs_queue, timestamp, 0, lo, hi)

                if lo1 == hi1:
                    jobs_queue.insert(lo1, new_job_id)

                elif lo1 != hi1:
                    # There are already entries with same importance and timestamp
                    # Check for duration comparison

                    # Runtime: O(logn)

                    lo2 = reverse_bisect_left(jobs_queue, duration, 5, lo1, hi1)
                    hi2 = reverse_bisect_right(jobs_queue, duration, 5, lo1, hi1)

                    # lo2 must be equal to hi2, due to unique triplet(imp,timestamp,duration) condition
                    jobs_queue.insert(lo2, new_job_id)

        else:
            # if jobs_queue is empty, just append the current job

            # Runtime: O(1)

            jobs_queue.append(new_job_id)

        jobs_history[timestamp] = list(jobs_queue)
        # Update job_id for next job
        new_job_id += 1

    elif parts[0] == 'assign':
        timestamp = int(parts[1])
        k = int(parts[2])
        if num_cpu == 0 or k == 0:
            continue
        jobs_processing = [job for job in jobs_processing if job>timestamp]

        if len(jobs_processing)<num_cpu:
            # Assigns the jobs to free CPUs in decreasing priority

            # Runtime: O(number of CPUs)

            jobs_to_be_processed = min(k, num_cpu- len(jobs_processing), len(jobs_queue))
            while jobs_to_be_processed > 0:
                # While loop to handle the edge case where jobs are of duration 0
                count = 0
                for job in reversed(jobs_queue[-jobs_to_be_processed:]):
                    print_job(job)
                    count += 1
                    jobs_processing.append(timestamp + master_job_dict[job][-1])
                k = k - count
                jobs_processing = [job for job in jobs_processing if job>timestamp]
                jobs_queue = jobs_queue[:-jobs_to_be_processed]

                jobs_to_be_processed = min(k, num_cpu-len(jobs_processing), len(jobs_queue))

            jobs_history[timestamp] = list(jobs_queue)

    elif parts[0] == 'query':
        timestamp = int(parts[1])
        # Sorting the timestamps in the jobs_history dictionary
        # Runtime: O(nlogn)
        history_timestamps = sorted(jobs_history.keys())
        if timestamp not in history_timestamps:
            # Bisection to find the nearest lowerbound timestamp
            # Runtime: O(logn)
            timestamp_index = bisect.bisect_left(history_timestamps, timestamp)
            if timestamp_index == 0:
                continue
            timestamp = history_timestamps[timestamp_index-1]

        queue_at_timestamp = jobs_history[timestamp]

        try:
            # This is a not an origin related query
            # Runtime: O(k)
            k = int(parts[2])
            if k == 0:
                continue
            if timestamp in jobs_history:

                if len(queue_at_timestamp) == 0:
                    continue
                else:
                    result = min(k, len(queue_at_timestamp))
                    for job in reversed(queue_at_timestamp[-result:]):
                        print_job(job)
        except ValueError:
            # This is a origin related query
            # Runtime: O(s at timestamp)
            orig = parts[2]

            result = [job for job in reversed(queue_at_timestamp) if orig_list[master_job_dict[job][2]] == orig]
            if len(result) > 0:
                for job in result:
                    print_job(job)

