import csv
import matplotlib.pyplot as plt


if __name__ == "__main__":
    path = "/mnt/gemini/data1/yifengliu/data/wmt-da/train.csv"
    score_list = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            score_list.append(float(row['raw']))
    
    save_path = "/mnt/gemini/data1/yifengliu/qe-lr/output/wmt_da_hist.png"
    plt.hist(score_list, bins=[i for i in range(0, 101, 1)], alpha=0.75, color='blue', edgecolor='black')
    plt.xlabel('Raw Score')
    plt.ylabel('Frequency')
    plt.title('Histogram of WMT DA Raw Scores')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(save_path)