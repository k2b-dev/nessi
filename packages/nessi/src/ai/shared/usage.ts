import type { Usage } from "../types.js";

export const makeUsage = (input = 0, output = 0): Usage =>
  ({ input, output, total: input + output });
