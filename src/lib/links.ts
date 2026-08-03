/**
 * Rewrite the relative documentation links copied verbatim from the vllm-ascend
 * Sphinx docs (docs/source/tutorials/models/*.md) into the recipe YAML content.
 *
 * Recipe markdown is treated as if it lived at docs/source/tutorials/models/
 * inside the vllm-ascend repo, so relative links resolve to GitHub blob URLs
 * at a pinned tag:
 *
 *   ../../installation.md                       -> .../blob/v0.23.0rc1/docs/source/installation.md
 *   ../../user_guide/configuration/env_vars.md  -> .../blob/v0.23.0rc1/docs/source/user_guide/configuration/env_vars.md
 *   ../features/pd_disaggregation_mooncake_multi_node.md
 *                                               -> .../blob/v0.23.0rc1/docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md
 *   https://... (absolute)  and  #anchor        -> returned unchanged
 */

export const VLLM_ASCEND_VERSION = 'v0.23.0rc1';

const VLLM_ASCEND_DOCS_BLOB_BASE = `https://github.com/vllm-project/vllm-ascend/blob/${VLLM_ASCEND_VERSION}/docs/source/`;

/** Recipe markdown originates from model tutorial pages in the vllm-ascend docs. */
const RECIPE_SOURCE_BASE = ['docs', 'source', 'tutorials', 'models'];

export function resolveVllmAscendLink(href: string): string {
  // Pass through anything that is not a repo-relative path:
  //   - in-page anchor (#foo)
  //   - root-relative (/foo)
  //   - any URI scheme (http:, https:, mailto:, tel:, ...)
  if (href.startsWith('#') || href.startsWith('/') || /^[a-z][a-z0-9+.-]*:/i.test(href)) {
    return href;
  }

  // Split the anchor fragment so it survives path resolution and is re-appended.
  const hashIndex = href.indexOf('#');
  const fragment = hashIndex === -1 ? '' : href.slice(hashIndex);
  const path = hashIndex === -1 ? href : href.slice(0, hashIndex);

  // Resolve ./ and ../ segments against docs/source/tutorials/models/.
  const segments = [...RECIPE_SOURCE_BASE];
  for (const segment of path.split('/')) {
    if (segment === '' || segment === '.') continue;
    if (segment === '..') segments.pop();
    else segments.push(segment);
  }

  const resolved = segments.join('/'); // e.g. docs/source/user_guide/configuration/env_vars.md
  const relative = resolved.startsWith('docs/source/')
    ? resolved.slice('docs/source/'.length)
    : resolved;
  return `${VLLM_ASCEND_DOCS_BLOB_BASE}${relative}${fragment}`;
}
