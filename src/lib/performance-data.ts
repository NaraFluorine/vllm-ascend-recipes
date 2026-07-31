import { existsSync } from 'node:fs';
import { resolve } from 'node:path';
import xlsx from 'xlsx';
import type { PerformanceTable } from './types';

type CellValue = string | number | boolean | Date | null | undefined;

const performanceFile = resolve(process.env.PERFORMANCE_XLSX_PATH ?? 'op.xlsx');
const { readFile, utils } = xlsx;
const modelHeader = '模型';

function logPerformance(event: string, details: Record<string, unknown> = {}) {
  console.info(`[performance-data] ${event}`, details);
}

function cellToString(value: CellValue): string {
  if (value === null || value === undefined) return '';
  if (value instanceof Date) return value.toISOString().slice(0, 10);
  return String(value).trim();
}

export function getPerformanceTables(modelNames: string[]): PerformanceTable[] {
  const fileExists = existsSync(performanceFile);
  logPerformance('lookup started', {
    cwd: process.cwd(),
    file: performanceFile,
    fileExists,
    configuredPath: process.env.PERFORMANCE_XLSX_PATH ?? null,
    modelNames,
  });

  if (!fileExists) {
    console.warn(
      `[performance-data] op.xlsx was not found at ${performanceFile}. ` +
        'The file must exist on the machine running Astro/build.',
    );
    return [];
  }

  if (modelNames.length === 0) {
    console.warn('[performance-data] no performance_model_names configured; skipping lookup');
    return [];
  }

  let workbook: xlsx.WorkBook;
  try {
    workbook = readFile(performanceFile, { cellDates: true });
  } catch (error) {
    console.error('[performance-data] failed to read workbook', {
      file: performanceFile,
      error: error instanceof Error ? error.message : String(error),
    });
    throw error;
  }

  logPerformance('workbook loaded', {
    file: performanceFile,
    sheetCount: workbook.SheetNames.length,
    sheets: workbook.SheetNames,
  });

  const acceptedModelNames = new Set(modelNames.map((name) => name.trim()).filter(Boolean));
  const tables: PerformanceTable[] = [];

  for (const sheetName of workbook.SheetNames) {
    const sheet = workbook.Sheets[sheetName];
    if (!sheet) {
      console.warn('[performance-data] sheet reference is missing', { sheetName });
      continue;
    }

    const rawRows = utils.sheet_to_json<CellValue[]>(sheet, { header: 1, defval: '' });
    const rows = rawRows
      .map((row) => row.map(cellToString))
      .filter((row) => row.some((cell) => cell.length > 0));

    if (rows.length < 2) {
      console.warn('[performance-data] sheet has no data rows', {
        sheetName,
        rowCount: rows.length,
      });
      continue;
    }

    if (rows[0][0] !== modelHeader) {
      console.warn('[performance-data] sheet skipped because first header is not 模型', {
        sheetName,
        firstHeader: rows[0][0] ?? '',
      });
      continue;
    }

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
      logPerformance('sheet matched', {
        sheetName,
        headerCount: headers.length,
        headers,
        matchedRowCount: dataRows.length,
        matchedModels: dataRows.map((row) => row[0]),
      });
    } else {
      logPerformance('sheet has no matching models', {
        sheetName,
        headerCount: headers.length,
        dataRowCount: rows.length - 1,
        acceptedModelNames: [...acceptedModelNames],
      });
    }
  }

  logPerformance('lookup finished', {
    tableCount: tables.length,
    totalMatchedRowCount: tables.reduce((count, table) => count + table.rows.length, 0),
  });

  return tables;
}
