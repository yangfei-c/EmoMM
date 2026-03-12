import subprocess
import sys

def run_experiment(i,script_path, loss_type, model_type, desc):
    if i==0:
        print(f"开始实验: [{desc}] (Model: {model_type} | Loss: {loss_type})")
    else:
        print(f"\n开始实验: [{desc}] (Model: {model_type} | Loss: {loss_type})")
    cmd = [
        sys.executable,
        script_path,
        f"loss_type={loss_type}",
        f"model_type={model_type}",
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ 完成: [{desc}]")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 中断: 运行出错，退出码: {e.returncode}")
        sys.exit(1)

def main():
    experiments = [
        ("train.py", "BCE", "MLP", "基础特征 + 传统二分类损失"),
        ("train.py", "ASL", "MLP", "基础特征 + 静态非对称损失"),
        ("train.py", "DynamASCL", "MLP", "证明自适应动态 Loss 对长尾分布的有效性"),
        ("train.py", "BCE", "Transformer", "证明高级时序架构需要匹配高级 Loss"),
        ("train.py", "DynamASCL", "Transformer", "时序注意力架构 + 动态非对称损失完全体")
    ]
    for i,(script, loss, model, desc) in enumerate(experiments):
        run_experiment(i,script, loss, model, desc)
    print(f"\n🎉 {len(experiments)} 组实验已全部跑完！")

if __name__ == "__main__":
   main()