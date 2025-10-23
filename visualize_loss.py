import argparse
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def parse_log(log_file):
    """
    解析训练日志文件，提取每个数据点对应的 Epoch 和损失值。
    """
    print(f"--- 开始调试: 解析日志文件 (模式: 按 Epoch) ---")
    print(f"[调试] 目标文件: {log_file}")
    records = []
    
    # 正则表达式，增加对 Epoch 的捕获
    epoch_pattern = re.compile(r"Epoch\[(\d+)\]")
    iter_pattern = re.compile(r"Iteration\[(\d+)/\d+\]")
    loss_patterns = {
        'id_loss': re.compile(r"id_loss: ([\d\.]+)"),
        'reg_loss': re.compile(r"reg_loss: ([\d\.]+)"),
        'sdm_loss': re.compile(r"sdm_loss: ([\d\.]+)"),
        'mlm_loss': re.compile(r"mlm_loss: ([\d\.]+)"),
    }

    with open(log_file, 'r') as f:
        for line in f:
            epoch_match = epoch_pattern.search(line)
            iter_match = iter_pattern.search(line)
            
            # 必须同时包含 Epoch 和 Iteration 才是一条有效的日志
            if epoch_match and iter_match:
                record = {'epoch': int(epoch_match.group(1))}
                
                for loss_name, pattern in loss_patterns.items():
                    loss_match = pattern.search(line)
                    if loss_match:
                        record[loss_name] = float(loss_match.group(1))
                    else:
                        record[loss_name] = np.nan
                
                records.append(record)

    if not records:
        print("[错误] 未能从日志中解析出任何数据记录。")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return df

def plot_loss_curves(df, output_path):
    """
    按 Epoch 绘制并保存损失曲线图，使用统一的Y轴。
    """
    if df.empty:
        print("No loss data found to plot.")
        return

    # --- 数据聚合：按 Epoch 计算平均值 ---
    df_epoch = df.groupby('epoch').mean().reset_index()
    print("\n[调试] 已按 Epoch 聚合数据，每个 Epoch 的平均损失 (前5行):")
    print(df_epoch.head())

    # --- 绘图设置 ---
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
    # 只创建一个坐标轴 ax
    fig, ax = plt.subplots(figsize=(18, 10))

    # --- 核心修改：在同一个坐标轴上绘制所有损失曲线 ---
    # 定义颜色和线型以便区分
    colors = {
        'id_loss': '#e41a1c',   # 红色
        'reg_loss': '#ff7f00',  # 橙色
        'sdm_loss': '#377eb8',  # 蓝色
        'mlm_loss': '#4daf4a'   # 绿色
    }
    linestyles = {
        'id_loss': '-',
        'reg_loss': '--',
        'sdm_loss': '-',
        'mlm_loss': ':'
    }
    labels = {
        'id_loss': 'ID Loss (Identity)',
        'reg_loss': 'Reg Loss (Identity)',
        'sdm_loss': 'SDM Loss (Instance)',
        'mlm_loss': 'MLM Loss (Instance)'
    }

    # 循环绘制所有损失
    for loss_name in ['id_loss', 'reg_loss', 'sdm_loss', 'mlm_loss']:
        if loss_name in df_epoch.columns and not df_epoch[loss_name].isnull().all():
            sns.lineplot(data=df_epoch, 
                         x='epoch', 
                         y=loss_name, 
                         ax=ax, 
                         label=labels[loss_name], 
                         color=colors[loss_name], 
                         linestyle=linestyles[loss_name],
                         linewidth=2.5, 
                         alpha=0.9)

    # --- 设置统一的坐标轴 ---
    ax.set_xlabel("Epoch", fontsize=16, weight='bold')
    ax.set_ylabel("Loss Value (Log Scale)", fontsize=16, weight='bold') # 统一的Y轴标签
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # 对数刻度，更好地展示初期剧烈变化和后期平稳状态
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="--", c='0.7')

    # --- 标题和图例 ---
    fig.suptitle("Training Loss Dynamics per Epoch (Unified Axis)", fontsize=22, weight='bold', y=0.98)
    
    # 从单个坐标轴获取图例
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, 
               loc='upper center', 
               bbox_to_anchor=(0.5, 0.92), 
               ncol=4, 
               fontsize=14,
               frameon=True,
               shadow=True,
               title='Loss Types',
               title_fontsize='13')
    ax.get_legend().remove() # 移除坐标轴内部的默认图例

    # 优化布局
    fig.tight_layout(rect=[0, 0.03, 1, 0.93]) # 调整布局以适应图例和标题
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Loss curve plot saved to: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize training losses from a log file.")
    parser.add_argument("log_file", type=str, help="Path to the training log file (e.g., train_log.txt).")
    args = parser.parse_args()

    log_dir = os.path.dirname(args.log_file)
    output_filename = "loss_visualization.png"
    output_path = os.path.join(log_dir, output_filename)

    df = parse_log(args.log_file)
    plot_loss_curves(df, output_path)

if __name__ == "__main__":
    main()