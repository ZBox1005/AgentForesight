"""
Math Trajectory Generation using AutoGen Swarm Framework.
Agents: MathSolver ↔ Verifier (Handoff-based Swarm pattern)

Uses OpenAI API (GPT-5.4-mini or GPT-4.1) via AutoGen's native OpenAIChatCompletionClient.

Usage:
    python -m collection.gen_math_autogen --dataset /path/to/MATH500 --num_tasks 2
"""
import asyncio
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import Swarm
from autogen_ext.models.openai import OpenAIChatCompletionClient


# ============================================================
# Math tools (registered as AutoGen FunctionTools)
# ============================================================

def compute(expression: str) -> str:
    """Execute Python/sympy code and return the result. Supports both single expressions
    and multi-line code. The last expression's value or any printed output is returned.

    All sympy functions and common variables are pre-loaded. You can freely use:
    - Math: sqrt, Rational, pi, oo, E, I, floor, ceil, ceiling, Abs, sign, factorial, binomial
    - Algebra: solve, simplify, expand, factor, factorint, collect, cancel, apart, together
    - Calculus: diff, integrate, limit, series, Sum, Product, summation
    - Trig: sin, cos, tan, asin, acos, atan, atan2, cot, sec, csc
    - Linear algebra: Matrix, det
    - Number theory: isprime, nextprime, divisors, totient, gcd, lcm, fibonacci, mod_inverse
    - Symbols: x,y,z,k,n,t,a,b,c,r,j,m,p,q,s,w,d,u,f,g,h (pre-defined)
    - Common SymPy helpers: sp (alias), solveset, interpolate, Point, div
    - You can also define new variables with symbols('v1 v2')

    Args:
        expression: Python/sympy code. Can be multi-line. Examples:
            "sqrt(9)"
            "solve(x**2-4, x)"
            "A = Matrix([[1,2],[3,4]]); A.det()"
            "from math import comb; comb(10,3)"
    """
    import sympy
    import math
    import keyword
    # Imported symbols enter the evaluation namespace through locals().
    from sympy import (symbols, solve, simplify, expand, factor, factorint, sqrt,
                       pi, oo, E, I, Rational, Integer, Float, S,
                       sin, cos, tan, asin, acos, atan, atan2, cot, sec, csc,
                       log, ln, exp, diff, integrate, limit, series,
                       Matrix, det, Transpose, Inverse, eye, zeros, ones, diag,
                       factorial, binomial, gcd, lcm, Abs, sign, floor, ceiling,
                       Sum, Product, summation, product,
                       Eq, Ne, Lt, Le, Gt, Ge, And, Or, Not,
                       Piecewise, Derivative, Integral, Limit,
                       isprime, nextprime, prevprime, divisors, totient,
                       nsolve, nsimplify, together, apart, cancel, collect, solveset,
                       interpolate, fibonacci, mod_inverse, Point,
                       trigsimp, powsimp, radsimp, combsimp, latex,
                       Poly, degree, roots, real_roots, nroots,
                       Function, Symbol, Wild, Dummy,
                       FiniteSet, Interval, Union, Intersection, EmptySet,
                       PolynomialError, sympify as sympify)
    from sympy.polys.polytools import div
    from sympy.abc import alpha, beta, gamma, delta, epsilon, theta, phi, psi, omega
    from math import comb, perm, ceil as math_ceil, isqrt

    # Pre-define common variable names (covers most agent usage patterns)
    x, y, z = symbols('x y z')
    k, n, t = symbols('k n t')
    a, b, c = symbols('a b c')
    r, j, m = symbols('r j m')
    p, q, s = symbols('p q s')
    w, d, u = symbols('w d u')
    f, g, h = symbols('f g h')
    A, B, C, D = symbols('A B C D')
    F, G, L, M, N, P, Q, R, T = symbols('F G L M N P Q R T')
    i, l, o, v = symbols('i l o v')

    indexed_names = " ".join(
        f"{prefix}{idx}"
        for prefix in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for idx in range(0, 51)
    )
    exec_locals = locals()
    exec_locals.update(dict(zip(indexed_names.split(), symbols(indexed_names))))
    exec_locals.update({
        "ceil": ceiling,
        "sp": sympy,
    })

    # Build execution namespace
    exec_ns = dict(exec_locals)
    exec_ns.update({"sympy": sympy, "math": math, "symbols": symbols,
                     "Symbol": Symbol, "Function": Function})

    def maybe_define_missing_name(err: Exception) -> bool:
        match = re.search(r"name '([A-Za-z][A-Za-z0-9]*)' is not defined", str(err))
        if not match:
            return False
        name = match.group(1)
        if keyword.iskeyword(name):
            return False
        if name in exec_ns:
            return False
        if re.fullmatch(r"[A-Za-z]", name) or re.fullmatch(r"[A-Za-z][0-9]+", name):
            exec_ns[name] = symbols(name)
            return True
        return False

    for _ in range(6):
        try:
            result = eval(expression, exec_ns)
            return str(result)
        except SyntaxError:
            break
        except Exception as e:
            if maybe_define_missing_name(e):
                continue
            if "invalid syntax" not in str(e):
                return f"Error: {e}"
            break

    # Multi-line: use exec with output capture
    for _ in range(6):
        try:
            import io, contextlib
            output = io.StringIO()
            clean_expr = expression.replace("sp.", "sympy.")
            clean_expr = clean_expr.replace("import sympy as sp", "")
            clean_expr = clean_expr.replace("import sympy", "")
            clean_expr = clean_expr.replace("from sympy import", "# from sympy import")
            clean_expr = clean_expr.replace("from math import", "# from math import")

            raw_lines = [
                line.rstrip()
                for line in clean_expr.strip().split("\n")
                if line.strip() and not line.lstrip().startswith("#")
            ]
            last_line = raw_lines[-1].lstrip() if raw_lines else ""
            if last_line and "=" not in last_line and not last_line.startswith(
                ("print", "for ", "if ", "while ", "def ", "class ", "with ", "try:", "except", "return")
            ):
                indent = raw_lines[-1][: len(raw_lines[-1]) - len(raw_lines[-1].lstrip())]
                raw_lines[-1] = f"{indent}print({last_line})"
                clean_expr = "\n".join(raw_lines)

            with contextlib.redirect_stdout(output):
                exec(clean_expr, exec_ns)

            printed = output.getvalue().strip()
            if printed:
                return printed

            for var_name in ["result", "answer", "ans", "res", "out", "output", "ret", "val"]:
                if var_name in exec_ns and exec_ns[var_name] is not None:
                    v = exec_ns[var_name]
                    if v != exec_ns.get(var_name + "_orig"):
                        return str(v)

            return "(executed successfully, no output)"
        except Exception as e:
            if maybe_define_missing_name(e):
                continue
            return f"Error: {e}"


def normalize_math_answer(text: str) -> str:
    s = str(text).strip()
    s = re.sub(r'\$', '', s)
    s = re.sub(r'\\boxed\{([^}]*)\}', r'\1', s)
    s = s.replace(r'\dfrac', r'\frac')
    for _ in range(3):
        s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', s)
    for _ in range(3):
        s = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', s)
    s = re.sub(r'\\sqrt(\d)', r'sqrt(\1)', s)
    s = re.sub(r'\\(?:text|mbox)\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\^?\\?circ', '', s)
    s = s.replace(r'\pi', 'pi')
    s = s.replace(r'\infty', 'oo')
    s = s.replace(r'\cup', 'U')
    s = s.replace(r'\pm', '+-')
    m_mat = re.search(r'\\begin\{[pb]matrix\}(.+?)\\end\{[pb]matrix\}', s, re.DOTALL)
    if m_mat:
        nums = re.findall(r'-?\d+/\d+|-?\d+\.?\d*', m_mat.group(1))
        if nums:
            s = '(' + ','.join(nums) + ')'
    m_pymat = re.search(r'[Mm]atrix\(\[(.+)\]\)', s)
    if m_pymat:
        nums = re.findall(r'-?\d+/\d+|-?\d+\.?\d*', m_pymat.group(1))
        if nums:
            s = '(' + ','.join(nums) + ')'
    s = re.sub(r'\\(left|right|mathrm|cdot|times)', '', s)
    s = re.sub(r'[{}\\]', '', s)
    s = s.replace('*', '')
    s = re.sub(r'\s*(cm|mm|m|km|kg|g|lb|ft|in|sec|min|hr|degrees?)$', '', s)
    s = re.sub(r'^[a-z]\s*=\s*', '', s)
    return s.replace(' ', '').lower().strip().rstrip('.')


def math_answers_equivalent(pred: str, gold: str) -> bool:
    if not pred or not gold:
        return False

    pn, gn = normalize_math_answer(pred), normalize_math_answer(gold)
    if pn == gn:
        return True

    try:
        import sympy

        pe = sympy.sympify(pn.replace('pi', str(sympy.pi)))
        ge = sympy.sympify(gn.replace('pi', str(sympy.pi)))
        if isinstance(pe, tuple) and isinstance(ge, tuple):
            if len(pe) == len(ge) and all(sympy.simplify(a - b) == 0 for a, b in zip(pe, ge)):
                return True
        elif sympy.simplify(pe - ge) == 0:
            return True
        # Numeric tolerance fallback for float vs fraction
        if pe.is_number and ge.is_number:
            if abs(float(pe) - float(ge)) < 1e-6:
                return True
    except Exception:
        pass

    pred_nums = re.findall(r'-?[\d.]+(?:/[\d.]+)?', pn)
    gold_nums = re.findall(r'-?[\d.]+(?:/[\d.]+)?', gn)
    if pred_nums and gold_nums and len(pred_nums) == len(gold_nums) and len(pred_nums) >= 1:
        try:
            from fractions import Fraction

            pvals = [float(Fraction(x)) for x in pred_nums]
            gvals = [float(Fraction(x)) for x in gold_nums]
            if all(abs(a - b) < 1e-6 for a, b in zip(pvals, gvals)):
                return True
        except Exception:
            pass

    if len(gold) >= 3 and gold.strip().lower() in pred.lower():
        return True
    return False


def verify_answer(expected: str, computed: str) -> str:
    """Verify if a computed answer matches the expected answer symbolically.

    Args:
        expected: The expected correct answer (string).
        computed: The computed answer to verify (string).
    """
    if math_answers_equivalent(computed, expected):
        return f"VERIFIED_CORRECT: {computed} equals {expected}"
    return f"VERIFIED_WRONG: {computed} ≠ {expected}"


# ============================================================
# Agent system prompts
# ============================================================

SOLVER_PROMPT = """You are MathSolver. You solve mathematical problems step by step.

RULES:
1. Break down the problem into clear steps.
2. Use the `compute` tool for concrete calculations, symbolic simplification, solving equations, and checking algebra.
3. Keep prose reasoning outside the tool. Do not send prose, Markdown, or natural language into `compute`.
4. If you need a fresh symbolic variable, define it explicitly with `symbols(...)` before using it.
5. Avoid unnecessary retries: if one tool call fails, fix the expression before trying again.
6. After solving, hand off to Verifier by using the handoff.
7. If Verifier finds an error and sends the problem back, fix your work based on their feedback.
"""

VERIFIER_PROMPT = """You are Verifier. You independently verify MathSolver's solutions.

RULES:
1. Review MathSolver's steps carefully, but recompute the final answer independently from the original problem.
2. Use `compute` for the independent recomputation and simplification.
3. Use `verify_answer` only after you have an independently recomputed answer.
4. In `verify_answer`, set `expected` to MathSolver's final answer and `computed` to your independently recomputed answer. Never pass the same expression as both arguments unless they truly came from independent work.
5. If CORRECT: respond with exactly "ANSWER_VERIFIED: <the_answer>" (this ends the task).
6. If WRONG: explain the error clearly and hand off back to MathSolver.
7. If a verification attempt fails because of a malformed tool expression, fix it once instead of looping on the same error.
"""


# ============================================================
# Team creation
# ============================================================

def create_model_client(api_key: str, model: str = "gpt-5.4-mini"):
    """Create OpenAI model client."""
    return OpenAIChatCompletionClient(
        model=model,
        api_key=api_key,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "openai",
            "structured_output": True,
        },
    )


def create_team(model_client, max_messages: int = 40):
    """Create AutoGen Swarm team: MathSolver ↔ Verifier."""
    solver = AssistantAgent(
        name="MathSolver",
        model_client=model_client,
        system_message=SOLVER_PROMPT,
        tools=[compute],
        handoffs=["Verifier"],
    )

    verifier = AssistantAgent(
        name="Verifier",
        model_client=model_client,
        system_message=VERIFIER_PROMPT,
        tools=[compute, verify_answer],
        handoffs=["MathSolver"],
    )

    termination = TextMentionTermination("ANSWER_VERIFIED") | MaxMessageTermination(max_messages)
    team = Swarm(participants=[solver, verifier], termination_condition=termination)
    return team


# ============================================================
# Trajectory extraction
# ============================================================

def extract_trajectory(task_result) -> list[dict]:
    """Convert AutoGen TaskResult messages to trajectory format."""
    steps = []
    for i, msg in enumerate(task_result.messages):
        step = {
            "step_number": i,
            "agent": getattr(msg, 'source', 'unknown'),
            "type": type(msg).__name__,
            "content": "",
        }

        # Extract content based on message type
        content = getattr(msg, 'content', '')
        if isinstance(content, str):
            step["content"] = content
        elif isinstance(content, list):
            calls = []
            results = []
            for item in content:
                if hasattr(item, 'arguments'):
                    # FunctionCall
                    calls.append({
                        "name": getattr(item, 'name', ''),
                        "arguments": item.arguments if isinstance(item.arguments, str) else json.dumps(item.arguments),
                        "id": getattr(item, 'id', ''),
                    })
                elif hasattr(item, 'call_id'):
                    # FunctionExecutionResult
                    results.append({
                        "name": getattr(item, 'name', ''),
                        "result": getattr(item, 'content', ''),
                        "call_id": getattr(item, 'call_id', ''),
                    })
            if calls:
                step["tool_calls"] = calls
                step["content"] = f"[Tool calls: {', '.join(c['name'] for c in calls)}]"
            elif results:
                step["tool_results"] = results
                step["content"] = "; ".join(f"{r['name']}={r['result']}" for r in results)
        else:
            step["content"] = str(content)

        steps.append(step)
    return steps


def extract_final_answer(task_result) -> str:
    """Extract ANSWER_VERIFIED: <answer> from the trajectory."""
    for msg in reversed(task_result.messages):
        content = str(getattr(msg, 'content', ''))
        m = re.search(r'ANSWER_VERIFIED:\s*(.+?)(?:\n|$)', content)
        if m:
            return m.group(1).strip()
    return ""


# ============================================================
# Main
# ============================================================

async def main():
    parser = argparse.ArgumentParser(description="Generate Math trajectories with AutoGen Swarm")
    parser.add_argument("--num_tasks", type=int, default=2)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="/tmp/math_traj")
    parser.add_argument("--model", type=str, default="gpt-5.4-mini")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to a datasets.load_from_disk MATH-500 directory with a test split")
    parser.add_argument("--max_messages", type=int, default=40)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: provide --api_key or set OPENAI_API_KEY")
        sys.exit(1)

    # Load dataset
    from datasets import load_from_disk
    ds = load_from_disk(args.dataset)["test"]
    tasks = [dict(ds[i]) for i in range(len(ds))]
    tasks = tasks[args.start_idx:args.start_idx + args.num_tasks]

    print(f"Model: {args.model}")
    print(f"Loaded {len(tasks)} tasks")
    print(f"Output: {args.output_dir}")

    model_client = create_model_client(api_key, args.model)

    results = []
    n_correct = 0

    for i, task in enumerate(tasks):
        question = task.get("problem", "")
        gold = str(task.get("answer", ""))
        meta = {k: task[k] for k in ["subject", "level", "unique_id"] if k in task}

        print(f"\n[{i+1}/{len(tasks)}] {meta.get('unique_id', '')}: {question[:80]}...")

        # Create fresh team per task
        team = create_team(model_client, args.max_messages)

        start = time.time()
        try:
            result = await team.run(
                task=f"Solve this math problem step by step using the compute tool.\n\nProblem: {question}"
            )
            trajectory = extract_trajectory(result)
            pred = extract_final_answer(result)
            stop_reason = result.stop_reason
        except Exception as e:
            import traceback
            traceback.print_exc()
            trajectory = []
            pred = ""
            stop_reason = f"error: {e}"

        elapsed = round(time.time() - start, 2)

        is_correct = math_answers_equivalent(pred, gold) if pred else False
        if is_correct:
            n_correct += 1

        agents = list(set(s["agent"] for s in trajectory))
        n_tools = sum(1 for s in trajectory if s.get("tool_calls"))

        record = {
            "task_idx": args.start_idx + i,
            "domain": "math",
            "task": question,
            "gold_answer": gold,
            "pred_answer": pred,
            "metadata": meta,
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "trajectory": trajectory,
            "num_steps": len(trajectory),
            "agents": agents,
            "num_tool_calls": n_tools,
            "stop_reason": str(stop_reason),
            "elapsed_s": elapsed,
            "is_correct": is_correct,
        }
        results.append(record)

        status = "✓" if is_correct else "✗"
        print(f"  {status} steps={len(trajectory)}, agents={agents}, tools={n_tools}, "
              f"pred='{pred[:40]}', gt='{gold[:20]}', time={elapsed}s")

        # Save incrementally
        outfile = os.path.join(args.output_dir,
                               f"math_autogen_{args.start_idx}_{args.start_idx+len(tasks)}.json")
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    await model_client.close()

    print(f"\n{'='*60}")
    print(f"SUMMARY: {n_correct}/{len(tasks)} correct ({100*n_correct/len(tasks):.0f}%)")
    print(f"Total tokens: {model_client.total_usage()}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
