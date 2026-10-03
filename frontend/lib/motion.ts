// Shared easing curve, typed as a literal tuple so it satisfies Motion's
// `Easing` type (a plain `number[]` is too wide and fails type-checking).
export const EASE_OUT = [0.16, 1, 0.3, 1] as const;
