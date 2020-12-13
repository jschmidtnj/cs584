import { configure, getLogger, Logger } from 'log4js';

export const initializeLogger = (): Logger => {
  configure({
    appenders: {
      console: { type: 'console' }
    },
    categories: {
      default: { appenders: ['console'], level: 'all' }
    }
  });
  const logger = getLogger();
  return logger;
};
