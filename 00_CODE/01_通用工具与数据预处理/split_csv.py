import os
import sys
import argparse
import logging
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import unittest


@dataclass
class SplitOperation:
    timestamp: str
    file_path: str
    original_lines: int
    new_files: List[str] = field(default_factory=list)
    status: str = "pending"
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


class CSVFileSplitter:
    DEFAULT_MAX_LINES = 750000

    def __init__(self, max_lines: int = None, log_level: str = "INFO", dry_run: bool = False):
        self.max_lines = max_lines or self.DEFAULT_MAX_LINES
        self.log_level = getattr(logging, log_level.upper())
        self.dry_run = dry_run
        self.operations: List[SplitOperation] = []
        self._setup_logger()

    def _setup_logger(self):
        self.logger = logging.getLogger("CSVFileSplitter")
        self.logger.setLevel(self.log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def count_lines(self, file_path: str) -> int:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

    def get_total_lines(self, file_path: str) -> int:
        return self.count_lines(file_path)

    def get_data_lines(self, file_path: str) -> int:
        return self.count_lines(file_path) - 1

    def split_file(self, file_path: str) -> SplitOperation:
        dir_name, file_name = os.path.split(file_path)
        file_base_name, file_ext = os.path.splitext(file_name)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        operation = SplitOperation(
            timestamp=timestamp,
            file_path=file_path,
            original_lines=self.get_data_lines(file_path)
        )

        if self.dry_run:
            self.logger.info(f"[DRY-RUN] 文件无需实际拆分: {file_path}")
            operation.status = "dry_run"
            self.operations.append(operation)
            return operation

        total_lines = self.get_data_lines(file_path)

        if total_lines <= self.max_lines:
            self.logger.info(f"文件行数未超过上限，无需拆分: {file_path}")
            operation.status = "skipped"
            self.operations.append(operation)
            return operation

        total_chunks = (total_lines + self.max_lines - 1) // self.max_lines
        self.logger.info(f"文件 {file_path} 需要拆分为 {total_chunks} 个部分 (总计 {total_lines} 行)")

        new_files = []
        try:
            with open(file_path, 'r', encoding='utf-8') as src:
                header = src.readline()
                part = 1
                line_count = 0
                out_file = None

                for line in src:
                    if line_count == 0:
                        split_file_name = f"{file_base_name}_{part}{file_ext}"
                        split_file_path = os.path.join(dir_name, split_file_name)
                        out_file = open(split_file_path, 'w', encoding='utf-8')
                        out_file.write(header)
                        new_files.append(split_file_path)
                        self.logger.info(f"  创建子文件: {split_file_name}")

                    out_file.write(line)
                    line_count += 1

                    if line_count >= self.max_lines:
                        out_file.close()
                        part += 1
                        line_count = 0

                if out_file and not out_file.closed:
                    out_file.close()

            os.remove(file_path)
            self.logger.info(f"已删除源文件: {file_path}")

            operation.new_files = new_files
            operation.status = "success"
            self.logger.info(f"拆分完成: {os.path.basename(file_path)} -> {len(new_files)} 个文件")

        except Exception as e:
            operation.status = "failed"
            operation.error = str(e)
            self.logger.error(f"拆分失败: {file_path}, 错误: {e}")

        self.operations.append(operation)
        return operation

    def process_directory(self, directory: str, file_type_filter: str = None) -> List[SplitOperation]:
        file_list = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".csv") and "part" not in file:
                    if file_type_filter and file_type_filter not in file:
                        continue
                    file_path = os.path.join(root, file)
                    file_list.append(file_path)

        total_files = len(file_list)
        self.logger.info(f"找到 {total_files} 个需要检查的 CSV 文件")

        for i, file_path in enumerate(file_list, 1):
            data_lines = self.get_data_lines(file_path)
            self.logger.debug(f"[{i}/{total_files}] 检查文件: {os.path.basename(file_path)} (行数: {data_lines})")

            if data_lines > self.max_lines:
                self.logger.info(f"[{i}/{total_files}] 文件超过上限，需要拆分: {file_path} ({data_lines} 行)")
                self.split_file(file_path)
            else:
                self.logger.debug(f"[{i}/{total_files}] 文件正常: {file_path} ({data_lines} 行)")

        return self.operations

    def check_npc_files(self, directory: str) -> dict:
        results = {
            "oversize_files": [],
            "non_npc_oversize": []
        }

        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".csv") and "part" not in file:
                    file_path = os.path.join(root, file)
                    data_lines = self.get_data_lines(file_path)

                    if data_lines > self.max_lines:
                        is_npc = "NPC_Simulation_Info" in file
                        file_info = {
                            "path": file_path,
                            "lines": data_lines,
                            "is_npc": is_npc
                        }
                        results["oversize_files"].append(file_info)

                        if not is_npc:
                            results["non_npc_oversize"].append(file_info)

        return results

    def generate_report(self) -> str:
        report = {
            "report_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "max_lines_threshold": self.max_lines,
            "dry_run": self.dry_run,
            "total_operations": len(self.operations),
            "operations": [op.to_dict() for op in self.operations]
        }
        return json.dumps(report, ensure_ascii=False, indent=2)


def check_all_files_type(directory: str, max_lines: int = 750000) -> dict:
    splitter = CSVFileSplitter(max_lines=max_lines)
    results = splitter.check_npc_files(directory)

    print("\n" + "=" * 60)
    print("超行数文件类型检查报告")
    print("=" * 60)

    if not results["oversize_files"]:
        print("未发现超过上限行数的文件")
        return results

    print(f"\n共发现 {len(results['oversize_files'])} 个超行数文件:")
    print("-" * 60)

    for i, file_info in enumerate(results["oversize_files"], 1):
        file_type = "NPC_Simulation_Info" if file_info["is_npc"] else "其他类型"
        status = "✓" if file_info["is_npc"] else "✗"
        print(f"{i}. {status} {file_type}")
        print(f"   路径: {file_info['path']}")
        print(f"   行数: {file_info['lines']}")
        print()

    if results["non_npc_oversize"]:
        print("警告: 以下文件不是 NPC_Simulation_Info 类型但超过行数上限:")
        for file_info in results["non_npc_oversize"]:
            print(f"  - {file_info['path']}")
    else:
        print("✓ 所有超行数文件均为 NPC_Simulation_Info 类型")

    print("=" * 60)
    return results


def main():
    parser = argparse.ArgumentParser(
        description='CSV 文件拆分工具 - 将超大 CSV 文件按行数拆分为多个子文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  python split_csv.py                           # 使用默认参数扫描 RAW_DATA 原始数据目录
  python split_csv.py --max-lines 500000      # 设置行数上限为 50万
  python split_csv.py --dry-run               # 仅模拟运行，不实际拆分
  python split_csv.py --log-level DEBUG       # 设置日志级别为 DEBUG
  python split_csv.py --check-only            # 仅检查文件类型，不进行拆分
  python split_csv.py D:\\data                # 指定扫描目录
        '''
    )

    parser.add_argument(
        'directory',
        nargs='?',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "RAW_DATA"),
        help='要扫描的目录路径 (默认为工程根目录下的 RAW_DATA 原始数据目录)'
    )

    parser.add_argument(
        '--max-lines', '-m',
        type=int,
        default=CSVFileSplitter.DEFAULT_MAX_LINES,
        help=f'每个子文件的最大行数 (默认: {CSVFileSplitter.DEFAULT_MAX_LINES})'
    )

    parser.add_argument(
        '--log-level', '-l',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别 (默认: INFO)'
    )

    parser.add_argument(
        '--dry-run', '-d',
        action='store_true',
        help='仅模拟运行，不实际拆分文件'
    )

    parser.add_argument(
        '--check-only', '-c',
        action='store_true',
        help='仅检查超行数文件类型，不进行拆分'
    )

    parser.add_argument(
        '--report', '-r',
        action='store_true',
        help='生成 JSON 格式的处理报告'
    )

    parser.add_argument(
        '--file-type',
        default=None,
        help='指定要处理的文件类型 (如 NPC_Simulation_Info)'
    )

    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"CSV 文件拆分工具")
    print(f"{'=' * 60}")
    print(f"扫描目录: {args.directory}")
    print(f"行数上限: {args.max_lines}")
    print(f"日志级别: {args.log_level}")
    print(f"模拟运行: {'是' if args.dry_run else '否'}")
    print(f"{'=' * 60}\n")

    if args.check_only:
        check_all_files_type(args.directory, args.max_lines)
        return

    splitter = CSVFileSplitter(
        max_lines=args.max_lines,
        log_level=args.log_level,
        dry_run=args.dry_run
    )

    splitter.process_directory(args.directory, args.file_type)

    if args.report:
        print("\n" + "=" * 60)
        print("处理报告")
        print("=" * 60)
        print(splitter.generate_report())
        print("=" * 60)

    print("\n程序执行完毕！")


if __name__ == "__main__":
    main()
