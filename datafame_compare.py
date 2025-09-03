#import packages, generate data and show top 5 rows
import pandas as pd
import numpy as np
from faker import Faker
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text
from IPython.display import display
from datetime import datetime
from rich.rule import Rule

console = Console()

#region session setup code
pd.set_option("display.max_columns", None)       # Show all columns
pd.set_option("display.width", 0)                # Use full terminal width (0 disables wrapping)
pd.set_option("display.max_colwidth", None)      # Show full content in each column
pd.set_option("display.expand_frame_repr", False) # Prevent wrapping to multiple lines
pd.set_option("display.max_rows", None)
#endregion

console = Console()

class DataframeCompare:
    def __init__(self, df1, df2, sorted_by=None,
                 tolerance=1e-6, within_6dp=True,
                 ignore_case=False, output="rich",
                 check_dtypes=True, show_all=False,
                 max_rows=20):

        self.sorted_by = sorted_by or []

        # Sort if sorted_by is provided
        if self.sorted_by:
            df1 = df1.sort_values(by=self.sorted_by).reset_index(drop=True)
            df2 = df2.sort_values(by=self.sorted_by).reset_index(drop=True)
        else:
            df1 = df1.reset_index(drop=True)
            df2 = df2.reset_index(drop=True)

        self.df1 = df1
        self.df2 = df2
        self.tolerance = tolerance
        self.within_6dp = within_6dp
        self.ignore_case = ignore_case
        self.output = output
        self.check_dtypes = check_dtypes
        self.show_all = show_all
        self.max_rows = max_rows

    def compare_values(self, compare_columns="ALL", return_structured=False):
        if compare_columns == "ALL":
            compare_cols = [col for col in self.df1.columns if col in self.df2.columns]
        elif isinstance(compare_columns, list):
            compare_cols = list(set(compare_columns + self.sorted_by))
            compare_cols = [col for col in compare_cols if col in self.df1.columns and col in self.df2.columns]
        else:
            raise ValueError("compare_columns must be 'ALL' or a list of column names")

        mismatches = []
        tolerance = self.tolerance if self.within_6dp else 0

        for col in compare_cols:
            for idx in range(len(self.df1)):
                val1 = self.df1.at[idx, col]
                val2 = self.df2.at[idx, col]

                if pd.isna(val1) and pd.isna(val2):
                    continue
                elif pd.isna(val1) or pd.isna(val2):
                    mismatches.append(self._record_mismatch(idx, col, val1, val2))
                elif pd.api.types.is_numeric_dtype(self.df1[col]) and pd.api.types.is_numeric_dtype(self.df2[col]):
                    if isinstance(val1, (bool, np.bool_)) or isinstance(val2, (bool, np.bool_)):
                        if val1 != val2:
                            mismatches.append(self._record_mismatch(idx, col, val1, val2))
                    else:
                        try:
                            if abs(val1 - val2) > tolerance:
                                mismatches.append(self._record_mismatch(idx, col, val1, val2))
                        except TypeError:
                            if val1 != val2:
                                mismatches.append(self._record_mismatch(idx, col, val1, val2))
                elif isinstance(val1, str) and isinstance(val2, str) and self.ignore_case:
                    if val1.lower() != val2.lower():
                        mismatches.append(self._record_mismatch(idx, col, val1, val2))
                else:
                    if val1 != val2:
                        mismatches.append(self._record_mismatch(idx, col, val1, val2))

        status = len(mismatches) == 0

        if not return_structured:
            grouped = {}
            for mismatch in mismatches:
                col = mismatch["column"]
                grouped.setdefault(col, []).append(mismatch["row_index"])
            return {
                "status": status,
                "mismatches": [
                    {"column": col, "mismatched_rows": rows}
                    for col, rows in grouped.items()
                ]
            }
        
        df = pd.DataFrame(mismatches)

        if status:
            return {
                "status": True,
                "message": "✅ All values match. No mismatches found.",
                "table": None,
                "mismatches": []
            }

        if self.output == "rich" and not df.empty:
            # First table: original order
            table1 = Table(title="🔍 Mismatches by Row Index", show_header=True, header_style="bright_white", box=box.SIMPLE)
            table1.add_column("row_index", style="bright_cyan", justify="right")
            for key in self.sorted_by:
                table1.add_column(key, style="bright_white", justify="right")
            table1.add_column("column", style="bright_white")
            table1.add_column("df1_value", style="bright_cyan")
            table1.add_column("df2_value", style="bright_yellow")
            table1.add_column("diff", style="bright_green")

            for _, row in df.iterrows():
                table1.add_row(
                    str(row["row_index"]),
                    *[str(row[key]) for key in self.sorted_by],
                    str(row["column"]),
                    str(row["df1_value"]),
                    str(row["df2_value"]),
                    str(row.get("diff", ""))
                )

            # Second table: sorted by sorted_by field(s)
            df_sorted = df.sort_values(by=self.sorted_by)
            table2 = Table(title=f"🔍 Mismatches by {', '.join(self.sorted_by)}", show_header=True, header_style="bright_white", box=box.SIMPLE)
            table2.add_column("row_index", style="bright_cyan", justify="right")
            for key in self.sorted_by:
                table2.add_column(key, style="bright_white", justify="right")
            table2.add_column("column", style="bright_white")
            table2.add_column("df1_value", style="bright_cyan")
            table2.add_column("df2_value", style="bright_yellow")
            table2.add_column("diff", style="bright_green")

            for _, row in df_sorted.iterrows():
                table2.add_row(
                    str(row["row_index"]),
                    *[str(row[key]) for key in self.sorted_by],
                    str(row["column"]),
                    str(row["df1_value"]),
                    str(row["df2_value"]),
                    str(row.get("diff", ""))
                )

            console.print(table1)
            console.print(table2)
        else:
            df = pd.DataFrame(mismatches)
            if not self.show_all:
                df = df.head(self.max_rows)
            print("\nMismatch Details:")
            print(df[["row_index"] + self.sorted_by + ["column", "df1_value", "df2_value", "diff"]].to_string(index=False))

        return {
            "status": False,
            "message": f"🔷 Value mismatches found in {len(mismatches)} rows.",
            "table": df,
            "mismatches": mismatches
        }

    def _record_mismatch(self, idx, col, val1, val2):
        row = {
            "row_index": idx,
            "column": col,
            "df1_value": val1,
            "df2_value": val2
        }

        # Include sorted_by values for traceability
        for key in self.sorted_by:
            row[key] = self.df1.at[idx, key]

        # Only include diff if both values are numeric
        if isinstance(val1, (int, float, np.number)) and isinstance(val2, (int, float, np.number)):
            try:
                row["diff"] = round(val2 - val1, 6)
            except Exception:
                row["diff"] = None
        else:
            row["diff"] = ""

        return row

    def compare_ordering(self):
        if not self.sorted_by:
            return {"status": True, "mismatched_rows": []}

        mismatched_rows = []
        for idx, (row1, row2) in enumerate(zip(self.df1[self.sorted_by].values, self.df2[self.sorted_by].values)):
            if not all(row1 == row2):
                mismatched_rows.append(idx)

        return {
            "status": len(mismatched_rows) == 0,
            "mismatched_rows": mismatched_rows
        }

    def mismatch_summary(self, show_table=True, compare_columns="ALL", mismatches=None):
        if mismatches is None:
            result = self.compare_values(compare_columns=compare_columns, return_structured=True)
            mismatches = result["mismatches"]

        summary = {}
        for item in mismatches:
            col = item["column"]
            summary.setdefault(col, 0)
            summary[col] += 1

        if show_table and self.output == "rich":
            table = Table(title="Mismatch Summary", box=box.SIMPLE)
            table.add_column("Column")
            table.add_column("Mismatch Count", justify="right")
            for col, count in summary.items():
                table.add_row(col, str(count))
            console.print(table)
        elif show_table:
            print(pd.DataFrame(list(summary.items()), columns=["Column", "Mismatch Count"]))

        return summary

    def structure(self):
        df1_cols = list(self.df1.columns)
        df2_cols = list(self.df2.columns)

        column_count_match = len(df1_cols) == len(df2_cols)
        column_names_match = df1_cols == df2_cols

        missing_in_df2 = [col for col in df1_cols if col not in df2_cols]
        extra_in_df2 = [col for col in df2_cols if col not in df1_cols]

        dtypes_match = True
        dtype_mismatches = {}
        dtype_violations = {}

        if column_names_match and self.check_dtypes:
            dtypes_match = False
            for col in df1_cols:
                dtype1 = self.df1[col].dtype
                dtype2 = self.df2[col].dtype
                if dtype1 != dtype2:
                    dtype_mismatches[col] = (dtype1, dtype2)

                    # Scan for offending rows using true index
                    offending = []
                    for idx, val in self.df2[col].items():
                        try:
                            if pd.isna(val):
                                continue
                            if dtype1 == "datetime64[ns]":
                                pd.to_datetime(val)
                            elif dtype1 == "float64":
                                float(val)
                            elif dtype1 == "int64":
                                int(val)
                            elif dtype1 == "bool":
                                bool(val)
                            elif dtype1 == "object":
                                str(val)
                        except Exception:
                            offending.append((idx, val))
                        if len(offending) >= 3:
                            break

                    total_violations = sum(
                        1 for val in self.df2[col]
                        if not pd.isna(val) and isinstance(val, str) and dtype1 != "object"
                    )

                    dtype_violations[col] = {
                        "count": total_violations,
                        "examples": offending
                    }

            dtypes_match = len(dtype_mismatches) == 0

        summary = {
            "column_count": {
                "status": column_count_match,
                "details": f"{len(df1_cols)} vs {len(df2_cols)}"
            },
            "column_names": {
                "status": column_names_match,
                "details": {
                    "missing_in_df2": missing_in_df2,
                    "extra_in_df2": extra_in_df2
                } if not column_names_match else None
            },
            "dtypes": {
                "status": dtypes_match,
                "details": dtype_mismatches if not dtypes_match else None
            } if self.check_dtypes else None
        }

        panels = []
        for key, result in summary.items():
            if result is None:
                continue
            status = "✅" if result["status"] else "❌"
            line = f"{status} {key.replace('_', ' ').title()}"
            if not result["status"] and result["details"]:
                if isinstance(result["details"], str):
                    line += f" — {result['details']}"
                elif isinstance(result["details"], dict):
                    for subkey, subval in result["details"].items():
                        if subval:
                            line += f"\n   • {subkey.replace('_', ' ').title()}: {subval}"
            panels.append(line)

        # Add offending row details for dtype mismatches
        if self.check_dtypes and dtype_violations:
            for col, info in dtype_violations.items():
                line = f"❌ {col} — {info['count']} offending rows"
                for idx, val in info["examples"]:
                    line += f"\n   • Row {idx}: {val}"
                panels.append(line)

        compact = "\n".join(panels)
        return {
            "summary": summary,
            "panels": panels,
            "compact": compact
        }
    

    def run(self, compare_columns="ALL", max_rows=None):
        effective_tolerance = self.tolerance if self.within_6dp else 0
        effective_max_rows = max_rows if max_rows is not None else self.max_rows

        result = self.structure()
        summary = result["summary"]
        failed = [name for name in ["column_count", "column_names"] + (["dtypes"] if self.check_dtypes else []) if not summary[name]["status"]]

        if self.output == "rich":
            console.rule("[bold cyan]Dataframe Comparison Summary")
            console.print(result["compact"])
        else:
            print("Dataframe Comparison Summary")
            print(result["compact"])

        # ─────────────────────────────────────────────────────────────────────────────
        # Row Count Check — Halt if mismatch
        # ─────────────────────────────────────────────────────────────────────────────
        if len(self.df1) != len(self.df2):
            if self.output == "rich":
                console.print(f"❌ Row Count — {len(self.df1)} vs {len(self.df2)}")
                console.rule("[bright red]Structural Issues Detected — Halting Further Comparison")
            else:
                print(f"❌ Row Count — {len(self.df1)} vs {len(self.df2)}")
                print("Structural Issues Detected — Halting Further Comparison")
            return

        if failed:
            if self.output == "rich":
                console.rule("[bright red]Structural Issues Detected — Halting Further Comparison")
            else:
                print("Structural Issues Detected — Halting Further Comparison")
            return

        if self.sorted_by:
            sort_key = ", ".join(self.sorted_by)
            if self.output == "rich":
                console.print(f"🔍 Sorted by: [{sort_key}]")
            else:
                print(f"Sorted by: {sort_key}")
                
        ordering_result = self.compare_ordering()
        if self.output == "rich":
            console.rule("[bold green]Ordering Check")
            if ordering_result["status"]:
                console.print("✅ Ordering matches perfectly. No mismatched rows.")
            else:
                console.print(f"🔍 Ordering mismatches: {len(ordering_result['mismatched_rows'])} rows")
        else:
            print("Ordering Check")
            print(f"Ordering mismatches: {len(ordering_result['mismatched_rows'])} rows")

        values_result = self.compare_values(compare_columns=compare_columns, return_structured=True)
        
        if self.output == "rich":
            console.rule("[bold green]Value Comparison")
            console.print(values_result["message"])
        else:
            print("Value Comparison")
            print(values_result["message"])
            if not values_result["status"] and values_result["mismatches"]:
                df = pd.DataFrame(values_result["mismatches"])
                if not self.show_all:
                    df = df.head(effective_max_rows)
                print("\nMismatch Details:")
                print(df[["row_index", "column", "df1_value", "df2_value", "diff"]].to_string(index=False))

        if self.output == "rich":
            console.rule("[bold green]Mismatch Summary by Column")
        else:
            print("Mismatch Summary by Column")

        self.mismatch_summary(show_table=True, compare_columns=compare_columns, mismatches=values_result["mismatches"])

###################    DECORATOR    #####################################################  
#########################################################################################
def divider_symbols(func):
    def wrapper(*args, **kwargs):
        header = f"###############  {func.__name__}  ###############################"
        console.print(header, style="bold cyan")
        result = func(*args, **kwargs)
        console.print(Rule(style="bright_magenta"))
        console.print(f"# End of {func.__name__} — {datetime.now().strftime('%H:%M:%S')}", style="dim")
        console.print("\n" * 3)
        return result
    return wrapper

###########################   FUNCTIONS   #############################################
#######################################################################################
@divider_symbols
def basic_test():
    # basic test with 2 issues
    df2 = df1.copy()
    df2.loc[2, "surname"] = df2.loc[2, "surname"] + "A"
    df2.loc[2, "balance"] = df2.loc[2, "balance"] + 999
    dc = DataframeCompare(df1, df2, check_dtypes=False, sorted_by=["acc_no"], output="rich")
    dc.run()

##########################################################
@divider_symbols
def missing_columns():
    #missing columns
    df2_missing_col = df1.copy()
    df2_missing_col.drop(columns=["region"], inplace=True)  # Remove one column

    dc_missing_col = DataframeCompare(df1, df2_missing_col, check_dtypes=False, sorted_by=["acc_no"], output="rich")
    dc_missing_col.run()

##########################################################
@divider_symbols
def different_row_count():
    # different row count
    df2_row_mismatch = df1.copy()
    extra_row = df1.iloc[0].copy()
    extra_row["acc_no"] += 1  # Ensure unique sort key
    df2_row_mismatch = pd.concat([df2_row_mismatch, pd.DataFrame([extra_row])], ignore_index=True)

    dc_row_mismatch = DataframeCompare(df1, df2_row_mismatch, check_dtypes=False,  sorted_by=["acc_no"], output="rich")
    dc_row_mismatch.run()

##########################################################
@divider_symbols
def sorting_different():
    #sorting mismatches, although should be correct automatically
    df2_unsorted = df1.copy()
    df2_unsorted = df2_unsorted.sample(frac=1).reset_index(drop=True)  # Shuffle rows

    dc_unsorted = DataframeCompare(df1, df2_unsorted, check_dtypes=False,  sorted_by=["acc_no"], output="rich")
    dc_unsorted.run()

##########################################################
@divider_symbols
def randomized_mismatch_test():
    df2_rand = df1.copy()
    num_rows = len(df2_rand)
    num_to_modify = max(1, int(num_rows * 0.1))  # At least one row

    # Track which columns have been cast to object
    object_casted = set()

    # Randomly select unique row indices to modify
    modified_indices = np.random.choice(df2_rand.index, size=num_to_modify, replace=False)

    for idx in modified_indices:
        # Randomly choose how many columns to modify (1 to 3)
        num_cols_to_modify = np.random.randint(1, 4)
        cols_to_modify = np.random.choice(df2_rand.columns, size=num_cols_to_modify, replace=False)

        for col in cols_to_modify:
            if col == "surname":
                df2_rand.at[idx, col] = df2_rand.at[idx, col] + "X"
            elif col == "forename":
                df2_rand.at[idx, col] = fake.first_name()
            elif col == "balance":
                df2_rand.at[idx, col] = round(df2_rand.at[idx, col] + np.random.uniform(100, 1000), 2)
            elif col == "product_type":
                df2_rand.at[idx, col] = np.random.choice(["Fixed", "Tracker", "Offset", "Variable"])
            elif col == "region":
                df2_rand.at[idx, col] = np.random.choice(["Scotland", "Wales", "England", "Northern Ireland"])
            elif col == "start_date":
                # 50% chance to insert invalid date
                if np.random.rand() < 0.5:
                    if col not in object_casted:
                        df2_rand[col] = df2_rand[col].astype("object")
                        object_casted.add(col)
                    df2_rand.at[idx, col] = "not-a-date"
                else:
                    df2_rand.at[idx, col] = pd.to_datetime("2023-01-01") + pd.to_timedelta(np.random.randint(1, 1000), unit="D")
            elif col == "current_rate":
                df2_rand.at[idx, col] = round(df2_rand.at[idx, col] + np.random.uniform(0.1, 1.0), 2)
            elif col == "term_months":
                df2_rand.at[idx, col] = np.random.choice([12, 24, 36, 48, 60])
            elif col == "is_joint_account":
                df2_rand.at[idx, col] = not df2_rand.at[idx, col]
            elif col == "acc_no":
                continue # as we don't want to change a key field

    # print(df2_rand)  # Optional: inspect mutated DataFrame
    dc_random = DataframeCompare(df1, df2_rand, check_dtypes=False,  sorted_by=["acc_no"], output="rich")
    dc_random.run()

##########################################################
if __name__ == "__main__":

    fake = Faker()
    np.random.seed(42)
    num_recs = 500

    df1 = pd.DataFrame({
        'acc_no': np.random.randint(10000000, 99999999, size=num_recs),
        'forename': [fake.first_name() for _ in range(num_recs)],
        'surname': [fake.last_name() for _ in range(num_recs)],
        'balance': np.round(np.random.uniform(100.0, 100000.0, size=num_recs), 2),
        'term_months': np.random.choice([12, 24, 36, 48, 60], size=num_recs),
        'current_rate': np.round(np.random.uniform(0.5, 5.0, size=num_recs), 2),
        'product_type': np.random.choice(['Fixed', 'Tracker', 'Offset', 'Variable'], size=num_recs),
        'start_date': pd.to_datetime(np.random.choice(pd.date_range('2015-01-01', '2022-12-31'), size=num_recs)),
        'is_joint_account': np.random.choice([True, False], size=num_recs),
        'region': np.random.choice(['Scotland', 'Wales', 'England', 'Northern Ireland'], size=num_recs)
    })
    display(df1)

    # basic_test()
    # missing_columns()
    # different_row_count()
    # sorting_different()
    randomized_mismatch_test()

