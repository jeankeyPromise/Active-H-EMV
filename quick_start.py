#!/usr/bin/env python3
"""
快速启动脚本 - 用于验证环境配置和快速测试
"""
import os
import sys
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major != 3 or version.minor != 10:
        print(f"❌ Python版本错误: 需要Python 3.10，当前版本: {version.major}.{version.minor}")
        return False
    print(f"✅ Python版本正确: {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """检查关键依赖"""
    required_packages = [
        'langchain',
        'langchain_openai',
        'torch',
        'sentence_transformers',
        'openai',
        'yaml',
    ]
    missing = []
    for package in required_packages:
        try:
            if package == 'yaml':
                __import__('yaml')
            elif package == 'langchain_openai':
                __import__('langchain_openai')
            else:
                __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"❌ {package} 未安装")
            missing.append(package)
    
    return len(missing) == 0

def check_api_key():
    """检查API密钥"""
    openai_key = os.getenv('OPENAI_API_KEY')
    google_key = os.getenv('GOOGLE_API_KEY')
    custom_key = os.getenv('CUSTOM_API_KEY') or os.getenv('QWEN_API_KEY')
    custom_base_url = os.getenv('CUSTOM_API_BASE_URL') or os.getenv('QWEN_API_BASE_URL')
    
    if openai_key:
        print(f"✅ OPENAI_API_KEY 已设置 (长度: {len(openai_key)})")
    else:
        print("⚠️  OPENAI_API_KEY 未设置")
    
    if google_key:
        print(f"✅ GOOGLE_API_KEY 已设置 (长度: {len(google_key)})")
    else:
        print("ℹ️  GOOGLE_API_KEY 未设置 (使用Gemini模型时需要)")
    
    if custom_key:
        print(f"✅ 自定义API密钥已设置 (长度: {len(custom_key)})")
        if custom_base_url:
            print(f"✅ 自定义API URL已设置: {custom_base_url}")
        else:
            print("⚠️  自定义API URL未设置 (CUSTOM_API_BASE_URL 或 QWEN_API_BASE_URL)")
    else:
        print("ℹ️  自定义API密钥未设置 (使用qwen-plus等自定义模型时需要)")
    
    return openai_key is not None or google_key is not None or (custom_key is not None and custom_base_url is not None)

def check_data_files():
    """检查数据文件"""
    data_dir = Path('data')
    if not data_dir.exists():
        print("❌ data目录不存在")
        return False
    
    print("✅ data目录存在")
    
    # 检查各个数据集
    datasets = {
        'armarx_lt_mem': ['qa.json', '2024-a7a-merged-summary.pkl'],
        'teach': ['test_set_25.pkl'],
        'ego4d_long_qa': ['qa.json'],
    }
    
    for dataset, files in datasets.items():
        dataset_path = data_dir / dataset
        if dataset_path.exists():
            print(f"  ✅ {dataset} 目录存在")
            for file in files:
                file_path = dataset_path / file
                if file_path.exists():
                    print(f"    ✅ {file} 存在")
                else:
                    print(f"    ⚠️  {file} 不存在")
        else:
            print(f"  ⚠️  {dataset} 目录不存在")
    
    return True

def check_config_files():
    """检查配置文件"""
    config_dir = Path('llm_emv/config')
    if not config_dir.exists():
        print("❌ llm_emv/config目录不存在")
        return False
    
    print("✅ llm_emv/config目录存在")
    
    # 检查主要配置
    main_configs = [
        'teach/simplified/full.yaml',
        'armarx_lt_mem/full.yaml',
        'ego4d/full.yaml',
    ]
    
    for config in main_configs:
        config_path = config_dir / config
        if config_path.exists():
            print(f"  ✅ {config} 存在")
        else:
            print(f"  ⚠️  {config} 不存在")
    
    return True

def main():
    """主函数"""
    print("=" * 60)
    print("H-EMV 环境检查")
    print("=" * 60)
    print()
    
    checks = [
        ("Python版本", check_python_version),
        ("依赖包", check_dependencies),
        ("API密钥", check_api_key),
        ("数据文件", check_data_files),
        ("配置文件", check_config_files),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n检查 {name}:")
        print("-" * 40)
        result = check_func()
        results.append((name, result))
    
    print("\n" + "=" * 60)
    print("检查总结")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 所有检查通过！可以开始复现实验。")
        print("\n快速开始:")
        print("  1. 交互式使用: python -m llm_emv --config armarx_lt_mem/full")
        print("  2. 运行评估: python -m llm_emv.eval --help")
    else:
        print("⚠️  部分检查未通过，请根据上述提示修复问题。")
        print("\n参考文档: REPRODUCTION_GUIDE.md")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())

