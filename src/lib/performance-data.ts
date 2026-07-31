import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import xlsx from 'xlsx';
import type { PerformanceTable } from './types';

type CellValue = string | number | boolean | Date | null | undefined;

const performanceFile = resolve(process.cwd(), 'op.xlsx');
const { readFile, utils } = xlsx;
const modelHeader = '模型';

function cellToString(value: CellValue): string {
  if (value === null || value === undefined) return '';
  if (value instanceof Date) return value.toISOString().slice(0, 10);
  return String(value).trim();
}

export function getPerformanceTables(modelNames: string[]): PerformanceTable[] {
  if (!existsSync(performanceFile) || modelNames.length === 0) return [];

  const workbook = readFile(performanceFile, { cellDates: true });
  const acceptedModelNames = new Set(modelNames.map((name) => name.trim()).filter(Boolean));
  const tables: PerformanceTable[] = [];

  for (const sheetName of workbook.SheetNames) {
    const sheet = workbook.Sheets[sheetName];
    if (!sheet) continue;

    const rawRows = utils.sheet_to_json<CellValue[]>(sheet, { header: 1, defval: '' });
    const rows = rawRows
      .map((row) => row.map(cellToString))
      .filter((row) => row.some((cell) => cell.length > 0));

    if (rows.length < 2 || rows[0][0] !== modelHeader) continue;

    const headerRow = rows[0];
    let columnCount = headerRow.length;
    while (columnCount > 1 && !headerRow[columnCount - 1]) columnCount--;

    const headers = headerRow.slice(0, columnCount);
    headers[0] = modelHeader;

    const dataRows = rows
      .slice(1)
      .filter((row) => acceptedModelNames.has(row[0] ?? ''))
      .map((row) => headers.map((_, index) => row[index] ?? ''));

    if (dataRows.length > 0) {
      tables.push({ sheetName, headers, rows: dataRows });
    }
  }

  return tables;
}
