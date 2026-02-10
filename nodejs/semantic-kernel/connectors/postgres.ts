// PostgreSQL vector store connector implementation for Semantic Kernel
import { Pool, PoolConfig } from 'pg'
import {
  DistanceFunction,
  FieldTypes,
  IndexKind,
  VectorStoreCollectionDefinition,
  VectorStoreField,
} from '../data/vector'
import { createDefaultLogger } from '../utils/logger'

const logger = createDefaultLogger('PostgresConnector')

// region: Constants

export const DEFAULT_SCHEMA = 'public'
export const MAX_DIMENSIONALITY = 2000
export const DISTANCE_COLUMN_NAME = 'sk_pg_distance'

// Environment variables
export const PGHOST_ENV_VAR = 'PGHOST'
export const PGPORT_ENV_VAR = 'PGPORT'
export const PGDATABASE_ENV_VAR = 'PGDATABASE'
export const PGUSER_ENV_VAR = 'PGUSER'
export const PGPASSWORD_ENV_VAR = 'PGPASSWORD'
export const PGSSLMODE_ENV_VAR = 'PGSSLMODE'

export enum SearchType {
  VECTOR = 'vector',
}

const DISTANCE_FUNCTION_MAP_STRING: Record<DistanceFunction, string> = {
  [DistanceFunction.CosineSimilarity]: 'vector_cosine_ops',
  [DistanceFunction.CosineDistance]: 'vector_cosine_ops',
  [DistanceFunction.DotProduct]: 'vector_ip_ops',
  [DistanceFunction.EuclideanDistance]: 'vector_l2_ops',
  [DistanceFunction.EuclideanSquaredDistance]: 'vector_l2_ops',
  [DistanceFunction.ManhattanDistance]: 'vector_l1_ops',
  [DistanceFunction.HammingDistance]: 'bit_hamming_ops',
  [DistanceFunction.DEFAULT]: 'vector_cosine_ops',
}

const DISTANCE_FUNCTION_MAP_OPS: Record<DistanceFunction, string> = {
  [DistanceFunction.CosineDistance]: '<=>',
  [DistanceFunction.CosineSimilarity]: '<=>',
  [DistanceFunction.DotProduct]: '<#>',
  [DistanceFunction.EuclideanDistance]: '<->',
  [DistanceFunction.EuclideanSquaredDistance]: '<->',
  [DistanceFunction.ManhattanDistance]: '<+>',
  [DistanceFunction.DEFAULT]: '<=>',
  [DistanceFunction.HammingDistance]: '',
}

const INDEX_KIND_MAP: Record<IndexKind, string> = {
  [IndexKind.HNSW]: 'hnsw',
  [IndexKind.IVFFlat]: 'ivfflat',
  [IndexKind.Flat]: 'flat',
  [IndexKind.DiskANN]: 'diskann',
  [IndexKind.QuantizedFlat]: 'quantizedflat',
  [IndexKind.Dynamic]: 'dynamic',
  [IndexKind.DEFAULT]: 'hnsw',
}

// region: Helpers

/**
 * Convert a TypeScript type string to PostgreSQL data type
 */
function pythonTypeToPostgres(typeStr: string): string | null {
  const typeMapping: Record<string, string> = {
    string: 'TEXT',
    number: 'DOUBLE PRECISION',
    boolean: 'BOOLEAN',
    object: 'JSONB',
    Date: 'TIMESTAMP',
    Buffer: 'BYTEA',
    null: 'NULL',
  }

  // Check for array types
  const listPattern = /(?:Array<(.+)>|(.+)\[\])/i
  const match = listPattern.exec(typeStr)
  if (match) {
    const elementType = match[1] || match[2]
    const postgresElementType = pythonTypeToPostgres(elementType)
    return postgresElementType ? `${postgresElementType}[]` : null
  }

  // Check for dictionary/object types
  if (typeStr.toLowerCase().includes('record') || typeStr.toLowerCase() === 'object') {
    return 'JSONB'
  }

  return typeMapping[typeStr] || null
}

/**
 * Convert a database row to a dictionary
 */
function convertRowToDict(row: any, fields: Array<[string, VectorStoreField | null]>): Record<string, any> {
  const result: Record<string, any> = {}

  fields.forEach(([fieldName, field]) => {
    const value = row[fieldName]
    if (value === null || value === undefined) {
      result[fieldName] = null
    } else if (field && field.fieldType === FieldTypes.VECTOR && typeof value === 'string') {
      // Parse vector string to array
      result[fieldName] = JSON.parse(value)
    } else {
      result[fieldName] = value
    }
  })

  return result
}

/**
 * Convert a dictionary to a database row
 */
function convertDictToRow(record: Record<string, any>, fields: VectorStoreField[]): any[] {
  return fields.map((field) => {
    const value = record[field.storageName || field.name]
    if (value === null || value === undefined) {
      return null
    }
    if (typeof value === 'object' && !Array.isArray(value) && !(value instanceof Date)) {
      return JSON.stringify(value)
    }
    return value
  })
}

// region: Interfaces

export interface PostgresSettings {
  connectionString?: string
  host?: string
  port?: number
  database?: string
  user?: string
  password?: string
  sslMode?: string
  minPool?: number
  maxPool?: number
  defaultDimensionality?: number
  maxRowsPerTransaction?: number
}

export interface VectorSearchOptions {
  vectorPropertyName?: string
  includeVectors?: boolean
  top?: number
  skip?: number
  filter?: any
  includeTotalCount?: boolean
}

export interface VectorSearchResult<T> {
  record: T
  score?: number
}

export interface GetFilteredRecordOptions {
  filter?: any
}

// region: Settings

export class PostgresConfig {
  connectionString?: string
  host?: string
  port: number
  database?: string
  user?: string
  password?: string
  sslMode?: string
  minPool: number
  maxPool: number
  defaultDimensionality: number
  maxRowsPerTransaction: number

  constructor(settings?: PostgresSettings) {
    this.connectionString = settings?.connectionString || process.env.POSTGRES_CONNECTION_STRING
    this.host = settings?.host || process.env[PGHOST_ENV_VAR] || process.env.POSTGRES_HOST
    this.port = settings?.port || parseInt(process.env[PGPORT_ENV_VAR] || process.env.POSTGRES_PORT || '5432')
    this.database = settings?.database || process.env[PGDATABASE_ENV_VAR] || process.env.POSTGRES_DBNAME
    this.user = settings?.user || process.env[PGUSER_ENV_VAR] || process.env.POSTGRES_USER
    this.password = settings?.password || process.env[PGPASSWORD_ENV_VAR] || process.env.POSTGRES_PASSWORD
    this.sslMode = settings?.sslMode || process.env[PGSSLMODE_ENV_VAR] || process.env.POSTGRES_SSL_MODE
    this.minPool = settings?.minPool || parseInt(process.env.POSTGRES_MIN_POOL || '1')
    this.maxPool = settings?.maxPool || parseInt(process.env.POSTGRES_MAX_POOL || '5')
    this.defaultDimensionality =
      settings?.defaultDimensionality || parseInt(process.env.POSTGRES_DEFAULT_DIMENSIONALITY || '100')
    this.maxRowsPerTransaction =
      settings?.maxRowsPerTransaction || parseInt(process.env.POSTGRES_MAX_ROWS_PER_TRANSACTION || '1000')
  }

  getConnectionConfig(): PoolConfig {
    if (this.connectionString) {
      return {
        connectionString: this.connectionString,
        min: this.minPool,
        max: this.maxPool,
      }
    }

    return {
      host: this.host,
      port: this.port,
      database: this.database,
      user: this.user,
      password: this.password,
      ssl: this.sslMode === 'require' ? { rejectUnauthorized: false } : this.sslMode === 'disable' ? false : undefined,
      min: this.minPool,
      max: this.maxPool,
    }
  }

  async createConnectionPool(): Promise<Pool> {
    try {
      const pool = new Pool(this.getConnectionConfig())
      // Test connection
      const client = await pool.connect()
      client.release()
      return pool
    } catch (error) {
      throw new Error(`Error creating connection pool: ${error}`)
    }
  }
}

// region: Collection

export class PostgresCollection<TKey extends string | number, TModel extends Record<string, any>> {
  connectionPool?: Pool
  dbSchema: string = DEFAULT_SCHEMA
  collectionName: string
  definition: VectorStoreCollectionDefinition
  recordType: new () => TModel
  managedClient: boolean
  private settings: PostgresConfig
  private distanceColumnName: string = DISTANCE_COLUMN_NAME

  constructor(options: {
    recordType: new () => TModel
    definition?: VectorStoreCollectionDefinition
    collectionName?: string
    connectionPool?: Pool
    dbSchema?: string
    settings?: PostgresSettings
  }) {
    this.recordType = options.recordType
    this.collectionName = options.collectionName || new options.recordType().constructor.name
    this.connectionPool = options.connectionPool
    this.dbSchema = options.dbSchema || DEFAULT_SCHEMA
    this.managedClient = !options.connectionPool
    this.settings = new PostgresConfig(options.settings)

    // Initialize definition
    if (options.definition) {
      this.definition = options.definition
    } else {
      // Create basic definition from record type
      this.definition = new VectorStoreCollectionDefinition({
        fields: [],
        collectionName: this.collectionName,
      })
    }

    this.initializeDistanceColumnName()
    this.validateDataModel()
  }

  private initializeDistanceColumnName(): void {
    let distanceColumnName = DISTANCE_COLUMN_NAME
    const storageNames = this.definition.fields.map((f) => f.storageName || f.name)
    let tries = 0

    while (storageNames.includes(distanceColumnName)) {
      const suffix = Math.random().toString(36).substring(2, 10)
      distanceColumnName = `${DISTANCE_COLUMN_NAME}_${suffix}`
      tries++
      if (tries > 10) {
        throw new Error('Unable to generate a unique distance column name.')
      }
    }

    this.distanceColumnName = distanceColumnName
  }

  async connect(): Promise<PostgresCollection<TKey, TModel>> {
    if (!this.connectionPool) {
      this.connectionPool = await this.settings.createConnectionPool()
    }
    return this
  }

  async close(): Promise<void> {
    if (this.managedClient && this.connectionPool) {
      await this.connectionPool.end()
      if (this.managedClient) {
        this.connectionPool = undefined
      }
    }
  }

  private validateDataModel(): void {
    for (const field of this.definition.fields) {
      if (field.fieldType === FieldTypes.VECTOR && field.dimensions && field.dimensions > MAX_DIMENSIONALITY) {
        throw new Error(
          `Dimensionality of ${field.dimensions} exceeds the maximum allowed value of ${MAX_DIMENSIONALITY}.`
        )
      }
    }
  }

  /**
   * Upsert records into the database
   */
  async upsert(records: TModel[]): Promise<TKey[]> {
    if (!this.connectionPool) {
      throw new Error('Connection pool is not available, please call connect() first.')
    }

    const keys: TKey[] = []
    const client = await this.connectionPool.connect()

    try {
      await client.query('BEGIN')

      const maxRows = this.settings.maxRowsPerTransaction
      for (let i = 0; i < records.length; i += maxRows) {
        const batch = records.slice(i, i + maxRows)
        const fields = this.definition.fields

        for (const record of batch) {
          const values = convertDictToRow(record, fields)
          const columnNames = fields.map((f) => `"${f.storageName || f.name}"`).join(', ')
          const placeholders = fields.map((_, idx) => `$${idx + 1}`).join(', ')
          const keyField = this.definition.fields.find((f) => f.fieldType === FieldTypes.KEY)
          const keyFieldName = keyField?.storageName || keyField?.name || 'id'

          const updateColumns = fields
            .filter((f) => f.name !== this.definition.keyName)
            .map((f) => {
              const name = f.storageName || f.name
              return `"${name}" = EXCLUDED."${name}"`
            })
            .join(', ')

          const query = `
            INSERT INTO "${this.dbSchema}"."${this.collectionName}" (${columnNames})
            VALUES (${placeholders})
            ON CONFLICT ("${keyFieldName}") DO UPDATE SET ${updateColumns}
            RETURNING "${keyFieldName}"
          `

          const result = await client.query(query, values)
          keys.push(result.rows[0][keyFieldName] as TKey)
        }
      }

      await client.query('COMMIT')
    } catch (error) {
      await client.query('ROLLBACK')
      throw error
    } finally {
      client.release()
    }

    return keys
  }

  /**
   * Get records by keys
   */
  async get(keys: TKey[], options?: GetFilteredRecordOptions): Promise<TModel[] | null> {
    if (!keys || keys.length === 0) {
      if (options) {
        throw new Error('Get without keys is not yet implemented.')
      }
      return null
    }

    if (!this.connectionPool) {
      throw new Error('Connection pool is not available, please call connect() first.')
    }

    const fields = this.definition.fields
    const selectList = fields.map((f) => `"${f.storageName || f.name}"`).join(', ')
    const keyField = fields.find((f) => f.fieldType === FieldTypes.KEY)
    const keyFieldName = keyField?.storageName || keyField?.name || 'id'

    const placeholders = keys.map((_, idx) => `$${idx + 1}`).join(', ')
    const query = `
      SELECT ${selectList}
      FROM "${this.dbSchema}"."${this.collectionName}"
      WHERE "${keyFieldName}" IN (${placeholders})
    `

    const result = await this.connectionPool.query(query, keys)

    if (!result.rows || result.rows.length === 0) {
      return null
    }

    const fieldTuples: Array<[string, VectorStoreField | null]> = fields.map((f) => [f.storageName || f.name, f])
    return result.rows.map((row) => convertRowToDict(row, fieldTuples) as TModel)
  }

  /**
   * Delete records by keys
   */
  async delete(keys: TKey[]): Promise<void> {
    if (!this.connectionPool) {
      throw new Error('Connection pool is not available, please call connect() first.')
    }

    const client = await this.connectionPool.connect()

    try {
      await client.query('BEGIN')

      const maxRows = this.settings.maxRowsPerTransaction
      const keyField = this.definition.fields.find((f) => f.fieldType === FieldTypes.KEY)
      const keyFieldName = keyField?.storageName || keyField?.name || 'id'

      for (let i = 0; i < keys.length; i += maxRows) {
        const batch = keys.slice(i, i + maxRows)
        const placeholders = batch.map((_, idx) => `$${idx + 1}`).join(', ')

        const query = `
          DELETE FROM "${this.dbSchema}"."${this.collectionName}"
          WHERE "${keyFieldName}" IN (${placeholders})
        `

        await client.query(query, batch)
      }

      await client.query('COMMIT')
    } catch (error) {
      await client.query('ROLLBACK')
      throw error
    } finally {
      client.release()
    }
  }

  /**
   * Ensure the collection (table) exists
   */
  async ensureCollectionExists(): Promise<void> {
    if (!this.connectionPool) {
      throw new Error('Connection pool is not available, please call connect() first.')
    }

    const columnDefinitions: string[] = []

    for (const field of this.definition.fields) {
      if (!field.type_) {
        throw new Error(`Property type is not defined for field '${field.name}'`)
      }

      const propertyType = pythonTypeToPostgres(field.type_) || field.type_.toUpperCase()
      const columnName = field.storageName || field.name

      if (field.fieldType === FieldTypes.VECTOR) {
        columnDefinitions.push(`"${columnName}" VECTOR(${field.dimensions})`)
      } else if (field.fieldType === FieldTypes.KEY) {
        columnDefinitions.push(`"${columnName}" ${propertyType} PRIMARY KEY`)
      } else {
        columnDefinitions.push(`"${columnName}" ${propertyType}`)
      }
    }

    const columnsStr = columnDefinitions.join(', ')
    const createTableQuery = `
      CREATE TABLE "${this.dbSchema}"."${this.collectionName}" (${columnsStr})
    `

    await this.connectionPool.query(createTableQuery)
    logger.info(`Postgres table '${this.collectionName}' created successfully.`)

    // Create indexes for vector fields
    for (const field of this.definition.fields) {
      if (field.fieldType === FieldTypes.VECTOR) {
        await this.createIndex(this.collectionName, field)
      }
    }
  }

  /**
   * Check if the collection exists
   */
  async collectionExists(): Promise<boolean> {
    if (!this.connectionPool) {
      throw new Error('Connection pool is not available, please call connect() first.')
    }

    const query = `
      SELECT table_name
      FROM information_schema.tables
      WHERE table_schema = $1 AND table_name = $2
    `

    const result = await this.connectionPool.query(query, [this.dbSchema, this.collectionName])
    return result.rows.length > 0
  }

  /**
   * Delete the collection (drop table)
   */
  async ensureCollectionDeleted(): Promise<void> {
    if (!this.connectionPool) {
      throw new Error('Connection pool is not available, please call connect() first.')
    }

    const query = `DROP TABLE "${this.dbSchema}"."${this.collectionName}" CASCADE`
    await this.connectionPool.query(query)
  }

  /**
   * Create an index on a vector column
   */
  private async createIndex(tableName: string, vectorField: VectorStoreField): Promise<void> {
    if (!this.connectionPool) {
      throw new Error('Connection pool is not available, please call connect() first.')
    }

    const distanceFunction = vectorField.distanceFunction || DistanceFunction.DEFAULT
    const indexKind = vectorField.indexKind || IndexKind.DEFAULT

    if (!(distanceFunction in DISTANCE_FUNCTION_MAP_STRING)) {
      throw new Error('Distance function must be set for HNSW indexes.')
    }

    if (!(indexKind in INDEX_KIND_MAP)) {
      throw new Error(`Index kind '${indexKind}' is not supported.`)
    }

    if (indexKind === IndexKind.IVFFlat && distanceFunction === DistanceFunction.ManhattanDistance) {
      throw new Error('IVF_FLAT index is not supported with MANHATTAN distance function.')
    }

    const columnName = vectorField.storageName || vectorField.name
    const indexName = `${tableName}_${columnName}_idx`
    const indexKindStr = INDEX_KIND_MAP[indexKind]
    const distanceOp = DISTANCE_FUNCTION_MAP_STRING[distanceFunction]

    const query = `
      CREATE INDEX "${indexName}"
      ON "${this.dbSchema}"."${tableName}"
      USING ${indexKindStr} ("${columnName}" ${distanceOp})
    `

    await this.connectionPool.query(query)
    logger.info(`Index '${indexName}' created successfully on column '${columnName}'.`)
  }

  /**
   * Vector similarity search
   */
  async search(
    vector: number[],
    options: VectorSearchOptions = {}
  ): Promise<{ results: VectorSearchResult<TModel>[]; totalCount?: number }> {
    if (!this.connectionPool) {
      throw new Error('Connection pool is not available, please call connect() first.')
    }

    const { query, params, returnFields } = this.constructVectorQuery(vector, options)

    if (options.includeTotalCount) {
      const result = await this.connectionPool.query(query, params)
      const rows = result.rows.map((row) => convertRowToDict(row, returnFields))
      const results = rows.map((row) => this.getVectorSearchResultFromRow(row))
      return { results, totalCount: results.length }
    } else {
      const result = await this.connectionPool.query(query, params)
      const rows = result.rows.map((row) => convertRowToDict(row, returnFields))
      const results = rows.map((row) => this.getVectorSearchResultFromRow(row))
      return { results }
    }
  }

  private constructVectorQuery(
    vector: number[],
    options: VectorSearchOptions
  ): { query: string; params: any[]; returnFields: Array<[string, VectorStoreField | null]> } {
    // Find the vector field
    const vectorField = this.definition.fields.find(
      (f) => f.fieldType === FieldTypes.VECTOR && (!options.vectorPropertyName || f.name === options.vectorPropertyName)
    )

    if (!vectorField) {
      throw new Error(`Vector field '${options.vectorPropertyName}' not found in the data model.`)
    }

    const distanceFunction = vectorField.distanceFunction || DistanceFunction.DEFAULT
    if (!(distanceFunction in DISTANCE_FUNCTION_MAP_OPS)) {
      throw new Error(`Distance function '${distanceFunction}' is not supported.`)
    }

    // Build select list
    const selectFields = this.definition.fields.filter(
      (f) => options.includeVectors || f.fieldType !== FieldTypes.VECTOR
    )
    const selectList = selectFields.map((f) => `"${f.storageName || f.name}"`).join(', ')
    const vectorColumnName = vectorField.storageName || vectorField.name
    const distanceOp = DISTANCE_FUNCTION_MAP_OPS[distanceFunction]

    let query = `
      SELECT ${selectList}, "${vectorColumnName}" ${distanceOp} $1 as "${this.distanceColumnName}"
      FROM "${this.dbSchema}"."${this.collectionName}"
    `

    // Add WHERE clause if filter exists
    if (options.filter) {
      // Filter implementation would go here
      // This is a simplified version
    }

    query += ` ORDER BY "${this.distanceColumnName}" LIMIT ${options.top || 10}`

    if (options.skip) {
      query += ` OFFSET ${options.skip}`
    }

    // Handle cosine similarity transformation
    if (distanceFunction === DistanceFunction.CosineSimilarity) {
      query = `
        SELECT subquery.*, 1 - subquery."${this.distanceColumnName}" AS "${this.distanceColumnName}"
        FROM (${query}) AS subquery
      `
    }

    // Handle dot product transformation
    if (distanceFunction === DistanceFunction.DotProduct) {
      query = `
        SELECT subquery.*, -1 * subquery."${this.distanceColumnName}" AS "${this.distanceColumnName}"
        FROM (${query}) AS subquery
      `
    }

    const vectorStr = '[' + vector.map((v) => String(v)).join(',') + ']'
    const params = [vectorStr]

    const returnFields: Array<[string, VectorStoreField | null]> = [
      ...selectFields.map((f): [string, VectorStoreField | null] => [f.storageName || f.name, f]),
      [this.distanceColumnName, null],
    ]

    return { query, params, returnFields }
  }

  private getVectorSearchResultFromRow(row: Record<string, any>): VectorSearchResult<TModel> {
    const score = row[this.distanceColumnName]
    const record = { ...row }
    delete record[this.distanceColumnName]

    return {
      record: record as TModel,
      score,
    }
  }
}

// region: Store

export class PostgresStore {
  connectionPool: Pool
  dbSchema: string = DEFAULT_SCHEMA
  tables?: string[]

  constructor(connectionPool: Pool, dbSchema: string = DEFAULT_SCHEMA, tables?: string[]) {
    this.connectionPool = connectionPool
    this.dbSchema = dbSchema
    this.tables = tables
  }

  async listCollectionNames(): Promise<string[]> {
    let query = `
      SELECT table_name
      FROM information_schema.tables
      WHERE table_schema = $1
    `
    const params: any[] = [this.dbSchema]

    if (this.tables && this.tables.length > 0) {
      const placeholders = this.tables.map((_, idx) => `$${idx + 2}`).join(', ')
      query += ` AND table_name IN (${placeholders})`
      params.push(...this.tables)
    }

    const result = await this.connectionPool.query(query, params)
    return result.rows.map((row) => row.table_name)
  }

  getCollection<TKey extends string | number, TModel extends Record<string, any>>(options: {
    recordType: new () => TModel
    definition?: VectorStoreCollectionDefinition
    collectionName?: string
  }): PostgresCollection<TKey, TModel> {
    return new PostgresCollection({
      ...options,
      connectionPool: this.connectionPool,
      dbSchema: this.dbSchema,
    })
  }
}
