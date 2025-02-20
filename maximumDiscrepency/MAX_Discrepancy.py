def read_scores_from_file(file):
    scores_dict = {}
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, score = line.split(',')
            score = float(score.strip())
            scores_dict[name] = score
    return scores_dict


def compute_differences(scores_dict1, scores_dict2):
    differences = []

    # 找相同的文件名并计算差异
    common_names = set(scores_dict1.keys()) & set(scores_dict2.keys())
    for name in common_names:
        score1 = scores_dict1[name]
        score2 = scores_dict2[name]
        difference = abs(score1 - score2)
        differences.append((name, difference))

    # 按差异降序排序
    differences.sort(key=lambda x: x[1], reverse=True)
    return differences


def main():
    file1 = 'destseg.txt'
    file2 = 'SimpleNet.txt'

    scores_dict1 = read_scores_from_file(file1)
    scores_dict2 = read_scores_from_file(file2)

    differences = compute_differences(scores_dict1, scores_dict2)

    # 输出前50个差异最大的项
    print("Top 25 differences:")
    for name, difference in differences[:28]:
        print(f"{name}: {difference:.10f}")
        #print(f"'dataset/{name}',")


if __name__ == "__main__":
    main()
