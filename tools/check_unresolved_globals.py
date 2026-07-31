from __future__ import annotations

import builtins
import symtable
from pathlib import Path

PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "market_cycle_trader_api"
)
IGNORED_GLOBALS = {"__file__", "__name__", "__package__"}
BUILTINS = set(dir(builtins))


def _module_defined(table: symtable.SymbolTable) -> set[str]:
    return {
        symbol.get_name()
        for symbol in table.get_symbols()
        if symbol.is_assigned()
        or symbol.is_imported()
        or symbol.is_namespace()
    }


def _referenced_globals(table: symtable.SymbolTable) -> set[str]:
    referenced: set[str] = set()
    for child in table.get_children():
        for symbol in child.get_symbols():
            if symbol.is_global() and symbol.is_referenced():
                referenced.add(symbol.get_name())
        referenced.update(_referenced_globals(child))
    return referenced


def main() -> int:
    failures: list[tuple[Path, list[str]]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        table = symtable.symtable(source, str(path), "exec")
        unresolved = sorted(
            _referenced_globals(table)
            - _module_defined(table)
            - BUILTINS
            - IGNORED_GLOBALS
        )
        if unresolved:
            failures.append((path.relative_to(PACKAGE_ROOT), unresolved))

    if failures:
        for path, names in failures:
            print(f"{path}: unresolved globals: {', '.join(names)}")
        return 1

    print("Python unresolved-global scan: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
