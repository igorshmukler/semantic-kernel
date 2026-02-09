import winston from 'winston'

// Logger configuration options
export interface LoggerOptions {
  level?: string
  format?: winston.Logform.Format
  transports?: winston.transport[]
  silent?: boolean
}

/**
 * Creates a default Winston logger for the Kernel
 */
export function createDefaultLogger(options?: LoggerOptions): winston.Logger {
  const defaultFormat = winston.format.combine(
    winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
    winston.format.errors({ stack: true }),
    winston.format.printf(({ level, message, timestamp, stack }) => {
      const baseMessage = `${timestamp} [${level.toUpperCase()}]: ${message}`
      return stack ? `${baseMessage}\n${stack}` : baseMessage
    })
  )

  return winston.createLogger({
    level: options?.level || 'info',
    format: options?.format || defaultFormat,
    transports: options?.transports || [
      new winston.transports.Console({
        format: winston.format.combine(winston.format.colorize(), defaultFormat),
      }),
    ],
    silent: options?.silent || false,
  })
}

export type Logger = winston.Logger
