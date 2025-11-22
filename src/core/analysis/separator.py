"""Separate Python code into Preprocessing, Inference, and Postprocessing blocks."""

import ast
from dataclasses import dataclass


@dataclass
class PipelineSegment:
    """Represents a segment of the pipeline (Pre/Inf/Post)."""

    body: list[ast.stmt]
    inputs: set[str]
    outputs: set[str]
    line_start: int
    line_end: int


@dataclass
class SeparatedPipeline:
    """Container for separated pipeline segments."""

    preprocess: PipelineSegment | None
    inference: PipelineSegment | None
    postprocess: PipelineSegment | None
    has_inference_marker: bool


class PipelineSeparator:
    """Analyzes Python AST to split code into Pre/Inf/Post stages."""

    def __init__(self):
        self.inference_marker = "# @inference"

    def separate(self, source_code: str, tree: ast.Module) -> SeparatedPipeline:
        """
        Split AST based on @inference comment marker.

        Args:
            source_code: Original Python source code
            tree: Parsed AST module

        Returns:
            SeparatedPipeline with Pre/Inf/Post segments
        """
        marker_line = self._find_inference_marker(source_code)

        if marker_line is None:
            print(f"⚠️  WARNING: No '{self.inference_marker}' marker found")
            print("   Treating entire code as Preprocessing block")
            return self._create_single_segment_pipeline(tree)

        # Check if marker is inside a function
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and (
                node.lineno <= marker_line <= (node.end_lineno or 999999)
            ):
                print(f"✓ Found '{self.inference_marker}' inside function: {node.name}")
                return self._split_function_body(node, source_code, marker_line)

        # Module-level separation (original behavior)
        return self._split_by_marker(tree, marker_line)

    def _find_inference_marker(self, source_code: str) -> int | None:
        """Find line number of @inference marker."""
        for i, line in enumerate(source_code.split("\n"), 1):
            if self.inference_marker in line:
                return i
        return None

    def _create_single_segment_pipeline(self, tree: ast.Module) -> SeparatedPipeline:
        """Create pipeline with single preprocessing segment."""
        segment = PipelineSegment(
            body=tree.body,
            inputs=set(),
            outputs=set(),
            line_start=1,
            line_end=len(tree.body),
        )
        self._track_variables(segment)

        return SeparatedPipeline(
            preprocess=segment,
            inference=None,
            postprocess=None,
            has_inference_marker=False,
        )

    def _split_by_marker(self, tree: ast.Module, marker_line: int) -> SeparatedPipeline:
        """Split AST body into Pre/Inf/Post based on marker line."""
        pre_body, inf_body, post_body = self._categorize_statements(
            tree.body, marker_line
        )
        pre_seg, inf_seg, post_seg = self._create_segments_from_bodies(
            pre_body, inf_body, post_body, marker_line
        )

        return SeparatedPipeline(
            preprocess=pre_seg if pre_body else None,
            inference=inf_seg if inf_body else None,
            postprocess=post_seg if post_body else None,
            has_inference_marker=True,
        )

    def _categorize_statements(
        self, statements: list, marker_line: int
    ) -> tuple[list, list, list]:
        """Categorize statements into pre/inf/post sections."""
        sections = {"pre": [], "inf": [], "post": []}
        current_section = "pre"
        inf_func_found = False

        for stmt in statements:
            stmt_line = getattr(stmt, "lineno", 0)

            if current_section == "pre" and stmt_line >= marker_line:
                current_section = "inf"

            if current_section == "inf" and isinstance(stmt, ast.FunctionDef):
                if inf_func_found:
                    current_section = "post"
                else:
                    inf_func_found = True

            sections[current_section].append(stmt)

        return sections["pre"], sections["inf"], sections["post"]

    def _create_segments_from_bodies(
        self, pre_body: list, inf_body: list, post_body: list, marker_line: int
    ) -> tuple:
        """Create pipeline segments from statement bodies."""
        pre_seg = self._create_segment(pre_body, 1, marker_line - 1)
        inf_seg = self._create_segment(
            inf_body, marker_line, marker_line + len(inf_body)
        )
        post_seg = self._create_segment(
            post_body,
            marker_line + len(inf_body) + 1,
            marker_line + len(inf_body) + len(post_body),
        )
        return pre_seg, inf_seg, post_seg

    def _find_postprocessing_marker(
        self, source_code: str, marker_line: int
    ) -> int | None:
        """Find postprocessing marker line after inference marker."""
        lines = source_code.split("\n")
        for i, line in enumerate(lines, 1):
            if i > marker_line and "# Postprocessing" in line:
                return i
        return None

    def _classify_statements(
        self, func_node: ast.FunctionDef, marker_line: int, post_marker_line: int | None
    ) -> tuple[list[ast.stmt], list[ast.stmt], list[ast.stmt]]:
        """Classify function statements into pre/inf/post sections."""
        sections = {"pre": [], "inf": [], "post": []}
        current_section = "pre"

        for stmt in func_node.body:
            if self._is_comment_stmt(stmt):
                continue

            stmt_line = getattr(stmt, "lineno", 0)
            current_section = self._determine_section(
                current_section,
                stmt_line,
                marker_line,
                post_marker_line,
                len(sections["inf"]),
            )
            sections[current_section].append(stmt)

        return sections["pre"], sections["inf"], sections["post"]

    def _determine_section(
        self,
        current: str,
        stmt_line: int,
        marker_line: int,
        post_marker_line: int | None,
        inf_count: int,
    ) -> str:
        """Determine which section a statement belongs to."""
        if current == "pre" and stmt_line >= marker_line:
            return "inf"

        if post_marker_line and stmt_line >= post_marker_line:
            return "post"

        if current == "inf" and inf_count >= 1:
            return "post"

        return current

    def _create_pipeline_segments(
        self,
        func_node: ast.FunctionDef,
        pre_stmts: list[ast.stmt],
        inf_stmts: list[ast.stmt],
        post_stmts: list[ast.stmt],
        marker_line: int,
    ) -> tuple[PipelineSegment | None, PipelineSegment | None, PipelineSegment | None]:
        """Create pipeline segments with proper input/output tracking."""
        func_params = [arg.arg for arg in func_node.args.args]

        # For preprocessing: use last defined variable as output
        pre_outputs = self._get_last_defined_var(pre_stmts, set(func_params))
        pre_seg = (
            self._create_function_segment_with_outputs(
                pre_stmts, func_params, pre_outputs, 1, marker_line - 1
            )
            if pre_stmts
            else None
        )

        inf_inputs = list(pre_seg.outputs) if pre_seg else func_params
        inf_outputs = self._find_used_variables(post_stmts, set())
        inf_seg = (
            self._create_function_segment_with_outputs(
                inf_stmts,
                inf_inputs,
                inf_outputs,
                marker_line,
                marker_line + len(inf_stmts),
            )
            if inf_stmts
            else None
        )

        post_inputs = list(inf_seg.outputs) if inf_seg else inf_inputs
        post_outputs = self._find_return_variables(func_node)
        post_seg = (
            self._create_function_segment_with_outputs(
                post_stmts,
                post_inputs,
                post_outputs,
                marker_line + len(inf_stmts) + 1,
                marker_line + len(inf_stmts) + len(post_stmts),
            )
            if post_stmts
            else None
        )

        return pre_seg, inf_seg, post_seg

    def _split_function_body(
        self, func_node: ast.FunctionDef, source_code: str, marker_line: int
    ) -> SeparatedPipeline:
        """Split function body into Pre/Inf/Post based on marker."""
        post_marker_line = self._find_postprocessing_marker(source_code, marker_line)

        pre_stmts, inf_stmts, post_stmts = self._classify_statements(
            func_node, marker_line, post_marker_line
        )

        pre_seg, inf_seg, post_seg = self._create_pipeline_segments(
            func_node, pre_stmts, inf_stmts, post_stmts, marker_line
        )

        return SeparatedPipeline(
            preprocess=pre_seg,
            inference=inf_seg,
            postprocess=post_seg,
            has_inference_marker=True,
        )

    def _is_comment_stmt(self, stmt: ast.stmt) -> bool:
        """Check if statement is a comment (Expr with Constant string)."""
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            return isinstance(stmt.value.value, str)
        return False

    def _create_function_segment(
        self, body: list[ast.stmt], inputs: list[str], start: int, end: int
    ) -> PipelineSegment | None:
        """Create segment from function body statements."""
        if not body:
            return None

        segment = PipelineSegment(
            body=body, inputs=set(inputs), outputs=set(), line_start=start, line_end=end
        )
        self._track_variables_from_inputs(segment, inputs)
        return segment

    def _get_last_defined_var(
        self, stmts: list[ast.stmt], exclude: set[str]
    ) -> set[str]:
        """Get the last defined variable in statements."""
        defined = set()
        for stmt in stmts:
            self._collect_defined_vars(stmt, defined)

        remaining = defined - exclude
        if remaining:
            for stmt in reversed(stmts):
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id in remaining:
                            return {target.id}
        return set()

    def _find_used_variables(
        self, stmts: list[ast.stmt], exclude: set[str]
    ) -> set[str]:
        """Find variables used in statements (but not defined in them)."""
        used = set()
        defined = set(exclude)

        for stmt in stmts:
            self._collect_used_vars(stmt, used, defined)
            self._collect_defined_vars(stmt, defined)

        return used

    def _find_return_variables(self, func_node: ast.FunctionDef) -> set[str]:
        """Find variables returned by function."""
        for stmt in reversed(func_node.body):
            if isinstance(stmt, ast.Return) and stmt.value:
                if isinstance(stmt.value, ast.Name):
                    return {stmt.value.id}
                elif isinstance(stmt.value, ast.Tuple):
                    vars_in_tuple = set()
                    for elt in stmt.value.elts:
                        if isinstance(elt, ast.Name):
                            vars_in_tuple.add(elt.id)
                    return vars_in_tuple
        return set()

    def _create_function_segment_with_outputs(
        self,
        body: list[ast.stmt],
        inputs: list[str],
        outputs: set[str],
        start: int,
        end: int,
    ) -> PipelineSegment | None:
        """Create segment with explicit inputs/outputs."""
        if not body:
            return None

        defined_vars = self._collect_all_defined_vars(body, inputs)
        final_outputs = self._determine_final_outputs(outputs, defined_vars, inputs)

        return PipelineSegment(
            body=body,
            inputs=set(inputs),
            outputs=final_outputs,
            line_start=start,
            line_end=end,
        )

    def _collect_all_defined_vars(
        self, body: list[ast.stmt], inputs: list[str]
    ) -> set[str]:
        """Collect all variables defined in body."""
        defined_vars = set(inputs)
        for stmt in body:
            self._collect_defined_vars(stmt, defined_vars)
        return defined_vars

    def _determine_final_outputs(
        self, outputs: set[str], defined_vars: set[str], inputs: list[str]
    ) -> set[str]:
        """Determine final output variables."""
        final_outputs = outputs & defined_vars
        if not final_outputs:
            remaining = defined_vars - set(inputs)
            final_outputs = {list(remaining)[-1]} if remaining else set()
        return final_outputs

    def _track_variables_from_inputs(self, segment: PipelineSegment, inputs: list[str]):
        """Track variables with known inputs."""
        defined_vars = set(inputs)
        used_vars = set()

        for stmt in segment.body:
            self._collect_used_vars(stmt, used_vars, defined_vars)
            self._collect_defined_vars(stmt, defined_vars)

        segment.outputs = defined_vars - set(inputs)
        for stmt in segment.body:
            if (
                isinstance(stmt, ast.Return)
                and stmt.value
                and isinstance(stmt.value, ast.Name)
            ):
                segment.outputs = {stmt.value.id}

    def _create_segment(
        self, body: list[ast.stmt], start: int, end: int
    ) -> PipelineSegment | None:
        """Create segment with variable tracking."""
        if not body:
            return None

        segment = PipelineSegment(
            body=body, inputs=set(), outputs=set(), line_start=start, line_end=end
        )
        self._track_variables(segment)
        return segment

    def _track_variables(self, segment: PipelineSegment):
        """Track input/output variables for segment."""
        defined_vars = set()
        used_vars = set()

        for stmt in segment.body:
            self._collect_used_vars(stmt, used_vars, defined_vars)
            self._collect_defined_vars(stmt, defined_vars)

        segment.inputs = used_vars - defined_vars
        segment.outputs = defined_vars

    def _collect_used_vars(self, node: ast.AST, used: set[str], defined: set[str]):
        """Collect variables used in node (before definition)."""
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Name)
                and isinstance(child.ctx, ast.Load)
                and child.id not in defined
            ):
                used.add(child.id)

    def _collect_defined_vars(self, node: ast.AST, defined: set[str]):
        """Collect variables defined in node."""
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined.add(node.target.id)
