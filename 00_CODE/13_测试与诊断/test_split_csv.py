import os
import sys
import unittest
import tempfile
import shutil
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_csv import CSVFileSplitter, SplitOperation, check_all_files_type


class TestCSVFileSplitter(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_dir2 = tempfile.mkdtemp()
        self.splitter = CSVFileSplitter(max_lines=100)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
        shutil.rmtree(self.test_dir2, ignore_errors=True)

    def create_test_csv(self, filename: str, line_count: int, header: str = "id,name\n") -> str:
        filepath = os.path.join(self.test_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(header)
            for i in range(line_count):
                f.write(f"{i},test_{i}\n")
        return filepath

    def test_count_lines(self):
        filepath = self.create_test_csv("test1.csv", 50)
        line_count = self.splitter.count_lines(filepath)
        self.assertEqual(line_count, 51)

    def test_get_data_lines(self):
        filepath = self.create_test_csv("test2.csv", 50)
        data_lines = self.splitter.get_data_lines(filepath)
        self.assertEqual(data_lines, 50)

    def test_split_small_file_skipped(self):
        filepath = self.create_test_csv("small.csv", 50)
        result = self.splitter.split_file(filepath)
        self.assertEqual(result.status, "skipped")
        self.assertTrue(os.path.exists(filepath))
        self.assertEqual(result.original_lines, 50)

    def test_split_large_file_success(self):
        filepath = self.create_test_csv("large.csv", 150)
        result = self.splitter.split_file(filepath)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.original_lines, 150)
        self.assertEqual(len(result.new_files), 2)
        self.assertFalse(os.path.exists(filepath))

        for part_file in result.new_files:
            self.assertTrue(os.path.exists(part_file))
            with open(part_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                self.assertGreater(len(lines), 1)
                self.assertEqual(lines[0], "id,name\n")

    def test_split_file_creates_correct_parts(self):
        filepath = self.create_test_csv("test_parts.csv", 250)
        result = self.splitter.split_file(filepath)
        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.new_files), 3)

        expected_names = ["test_parts_1.csv", "test_parts_2.csv", "test_parts_3.csv"]
        for expected in expected_names:
            expected_path = os.path.join(self.test_dir, expected)
            self.assertTrue(os.path.exists(expected_path))

    def test_split_operation_record(self):
        filepath = self.create_test_csv("record.csv", 200)
        result = self.splitter.split_file(filepath)
        self.assertIsNotNone(result.timestamp)
        self.assertEqual(result.file_path, filepath)
        self.assertEqual(result.original_lines, 200)
        self.assertEqual(len(self.splitter.operations), 1)

    def test_process_directory(self):
        self.create_test_csv("small1.csv", 50)
        self.create_test_csv("small2.csv", 30)
        self.create_test_csv("large1.csv", 150)
        self.create_test_csv("large2.csv", 200)

        self.splitter.process_directory(self.test_dir)
        self.assertEqual(len(self.splitter.operations), 4)

        success_ops = [op for op in self.splitter.operations if op.status == "success"]
        skipped_ops = [op for op in self.splitter.operations if op.status == "skipped"]
        self.assertEqual(len(success_ops), 2)
        self.assertEqual(len(skipped_ops), 2)

    def test_check_npc_files(self):
        self.create_test_csv("A5-1_NPC_Simulation_Info_test.csv", 150)
        self.create_test_csv("A5-1_EyeGaze_Simulation_Info_test.csv", 120)
        self.create_test_csv("A5-1_Log_Info_test.csv", 80)

        results = self.splitter.check_npc_files(self.test_dir)

        self.assertEqual(len(results["oversize_files"]), 3)
        self.assertEqual(len(results["non_npc_oversize"]), 2)

        npc_files = [f for f in results["oversize_files"] if f["is_npc"]]
        non_npc_files = [f for f in results["oversize_files"] if not f["is_npc"]]

        self.assertEqual(len(npc_files), 1)
        self.assertEqual(len(non_npc_files), 2)

    def test_dry_run_mode(self):
        filepath = self.create_test_csv("dryrun.csv", 150)
        dry_splitter = CSVFileSplitter(max_lines=100, dry_run=True)
        result = dry_splitter.split_file(filepath)

        self.assertEqual(result.status, "dry_run")
        self.assertTrue(os.path.exists(filepath))

    def test_check_all_files_type_function(self):
        self.create_test_csv("A5-1_NPC_Simulation_Info_test.csv", 150)
        self.create_test_csv("A5-1_Other_Info_test.csv", 120)

        results = check_all_files_type(self.test_dir, max_lines=100)

        self.assertEqual(len(results["oversize_files"]), 2)
        self.assertEqual(len(results["non_npc_oversize"]), 1)

    def test_header_preservation(self):
        custom_header = "col1,col2,col3\n"
        filepath = self.create_test_csv("header_test.csv", 150, header=custom_header)
        result = self.splitter.split_file(filepath)

        self.assertEqual(result.status, "success")
        for part_file in result.new_files:
            with open(part_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().rstrip('\n')
                self.assertEqual(first_line, "col1,col2,col3")

    def test_empty_file(self):
        filepath = os.path.join(self.test_dir, "empty.csv")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("id,name\n")

        result = self.splitter.split_file(filepath)
        self.assertEqual(result.status, "skipped")

    def test_generate_report(self):
        self.create_test_csv("report_test.csv", 150)
        self.splitter.split_file(self.create_test_csv("report_test.csv", 150))

        report = self.splitter.generate_report()
        self.assertIn("report_time", report)
        self.assertIn("operations", report)


class TestSplitOperation(unittest.TestCase):
    def test_split_operation_creation(self):
        op = SplitOperation(
            timestamp="2026-04-28 10:00:00",
            file_path="/test/path.csv",
            original_lines=1000
        )
        self.assertEqual(op.timestamp, "2026-04-28 10:00:00")
        self.assertEqual(op.file_path, "/test/path.csv")
        self.assertEqual(op.original_lines, 1000)
        self.assertEqual(op.status, "pending")
        self.assertEqual(op.new_files, [])

    def test_split_operation_to_dict(self):
        op = SplitOperation(
            timestamp="2026-04-28 10:00:00",
            file_path="/test/path.csv",
            original_lines=1000,
            new_files=["/test/path_1.csv", "/test/path_2.csv"],
            status="success"
        )
        d = op.to_dict()
        self.assertEqual(d["timestamp"], "2026-04-28 10:00:00")
        self.assertEqual(d["file_path"], "/test/path.csv")
        self.assertEqual(d["original_lines"], 1000)
        self.assertEqual(d["new_files"], ["/test/path_1.csv", "/test/path_2.csv"])
        self.assertEqual(d["status"], "success")


class TestFileNaming(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.splitter = CSVFileSplitter(max_lines=50)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_naming_with_timestamp(self):
        filepath = os.path.join(self.test_dir, "A5-1_NPC_Simulation_Info_2026_04_27_11_40_45.csv")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("id,name\n")
            for i in range(100):
                f.write(f"{i},test_{i}\n")

        result = self.splitter.split_file(filepath)
        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.new_files), 2)

        expected_parts = [
            "A5-1_NPC_Simulation_Info_2026_04_27_11_40_45_1.csv",
            "A5-1_NPC_Simulation_Info_2026_04_27_11_40_45_2.csv"
        ]
        for expected in expected_parts:
            expected_path = os.path.join(self.test_dir, expected)
            self.assertTrue(os.path.exists(expected_path), f"Expected {expected} to exist")


if __name__ == '__main__':
    unittest.main(verbosity=2)
