#!/usr/bin/env node
import { initializeLogger } from './utils';
import { getLogger } from 'log4js';
import finalhandler from 'finalhandler';
import { createServer } from 'http';
import serveIndex from 'serve-index';
import serveStatic from 'serve-static';
import { config } from 'dotenv';

const logger = getLogger();

let directory = 'static';
let PORT = 8000;


const main = (): void => {
  logger.info('starting presentation');

  if (process.argv.length > 2) {
    directory = process.argv[2];
  }

  if (process.env.PORT) {
    const givenPort = new Number(process.env.PORT);
    if (givenPort) {
      PORT = givenPort.valueOf();
    }
  }

  if (process.argv.length > 3) {
    const givenPort = new Number(process.argv[3]);
    if (givenPort) {
      PORT = givenPort.valueOf();
    }
  }

  const index = serveIndex(directory, {
    icons: true,
  });

  const serve = serveStatic(directory, {
    extensions: ['html'],
  });

  const server = createServer((req, res) => {
    const done = finalhandler(req, res);
    serve(req, res, () => {
      index(req as any, res as any, done);
    });
  });

  server.listen(PORT, () => {
    logger.info(`app is listening on http://localhost:${PORT} 🚀`);
  });
};

if (require.main === module) {
  config();
  const logger = initializeLogger();
  try {
    main();
  } catch (err) {
    logger.error((err as Error).message);
  }
}

export default main;
