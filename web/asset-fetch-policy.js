export const DEFAULT_ASSET_FETCH_ATTEMPTS_PER_SOURCE = 3;

export function buildAssetFetchAttemptPlan(
  sources,
  { attemptsPerSource = DEFAULT_ASSET_FETCH_ATTEMPTS_PER_SOURCE } = {},
) {
  if (!Array.isArray(sources) || sources.length === 0) {
    throw new TypeError("Asset fetch sources must be a non-empty array");
  }
  if (!Number.isInteger(attemptsPerSource) || attemptsPerSource < 1) {
    throw new TypeError("attemptsPerSource must be a positive integer");
  }
  const totalAttempts = sources.length * attemptsPerSource;
  return sources.flatMap((source, sourceIndex) => (
    Array.from({ length: attemptsPerSource }, (_, sourceAttemptIndex) => {
      const attemptIndex = sourceIndex * attemptsPerSource + sourceAttemptIndex;
      return {
        source,
        sourceIndex,
        sourceAttemptIndex,
        attemptIndex,
        attemptNumber: attemptIndex + 1,
        totalAttempts,
        cacheMode: sourceAttemptIndex === 0 ? "force-cache" : "reload",
      };
    })
  ));
}

export function assetFetchRetryDelayMs(sourceAttemptIndex, baseMs) {
  if (!Number.isInteger(sourceAttemptIndex) || sourceAttemptIndex < 0) {
    throw new TypeError("sourceAttemptIndex must be a non-negative integer");
  }
  if (!Number.isFinite(baseMs) || baseMs < 0) {
    throw new TypeError("baseMs must be a non-negative finite number");
  }
  return baseMs * (sourceAttemptIndex + 1);
}
