from collections import defaultdict

# Group sampled arguments by topic, and track their original list index
def group_args_by_topic_with_indices(args):
    grouped = defaultdict(list)
    for idx, arg in enumerate(args):
        arg['global_index'] = idx  # track position in original list
        grouped[arg['topic_id']].append(arg)
    return grouped

# Generate cyclic pairs within each topic group
def create_cyclic_pairs_within_topics(grouped_args, step):
    all_pairs = []
    for topic_id, args in grouped_args.items():
        n = len(args)
        if n < 2:
            continue  # skip topics with only one argument
        for i in range(n):
            for j in range(1, step + 1):
                a = args[i]['global_index']
                b = args[(i + j) % n]['global_index']
                pair = tuple(sorted((a, b)))
                if pair not in all_pairs:
                    all_pairs.append(pair)
    return all_pairs