import type { ScenarioScript } from './types';

const SCRIPT_REFERENCE_PATTERN = /\{\{script:([^{}]+)\}\}/g;

/** Expand scenario script references into fenced Markdown code blocks. */
export function expandScenarioScripts(
  content: string,
  scripts: Record<string, ScenarioScript> | undefined,
): string {
  const expanded = content.replace(SCRIPT_REFERENCE_PATTERN, (_reference, name: string) => {
    const script = scripts?.[name];
    if (!script) {
      throw new Error(`Unknown scenario script reference: ${name}`);
    }
    if (script.content.includes('{{script:')) {
      throw new Error(`Scenario script ${name} cannot reference another scenario script`);
    }

    return `\`\`\`${script.language}\n${script.content.trimEnd()}\n\`\`\``;
  });

  if (expanded.includes('{{script:')) {
    throw new Error('Malformed scenario script reference');
  }
  return expanded;
}
