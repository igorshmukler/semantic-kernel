/**
 * Recursively convert unhashable types to hashable equivalents.
 *
 * @param input - The input to convert to a hashable type
 * @param visited - A Map of visited objects to prevent infinite recursion
 * @returns The input converted to a hashable type
 */
export function makeHashable(input: any, visited?: Map<any, any>): any {
  if (visited === undefined) {
    visited = new Map()
  }

  // If we've seen this object before, return the stored placeholder or final result
  if (visited.has(input)) {
    return visited.get(input)
  }

  // Handle objects (equivalent to Pydantic models or dictionaries)
  if (input !== null && typeof input === 'object') {
    // Arrays, Sets, and other iterables
    if (Array.isArray(input) || input instanceof Set) {
      visited.set(input, null)
      const items = Array.from(input).map((item) => makeHashable(item, visited))
      const result = JSON.stringify(items)
      visited.set(input, result)
      return result
    }

    // Plain objects (dictionaries)
    if (input.constructor === Object || input.constructor?.name === 'Object') {
      visited.set(input, null)
      const entries = Object.entries(input)
        .map(([k, v]) => [k, makeHashable(v, visited)])
        .sort((a, b) => String(a[0]).localeCompare(String(b[0])))
      const result = JSON.stringify(entries)
      visited.set(input, result)
      return result
    }

    // Handle class instances with fields
    visited.set(input, null)
    const data: Record<string, any> = {}
    for (const key of Object.keys(input)) {
      if (!key.startsWith('_')) {
        // Skip private fields
        data[key] = makeHashable(input[key], visited)
      }
    }
    const entries = Object.entries(data).sort((a, b) => a[0].localeCompare(b[0]))
    const result = JSON.stringify(entries)
    visited.set(input, result)
    return result
  }

  // If it's already something hashable (primitive), just return it
  return input
}
