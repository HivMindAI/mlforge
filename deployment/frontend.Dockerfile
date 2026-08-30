FROM node:24-alpine AS dependencies

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

FROM node:24-alpine AS builder

ENV NEXT_TELEMETRY_DISABLED=1
ARG MLFORGE_API_ORIGIN=http://api:8000
ENV MLFORGE_API_ORIGIN=${MLFORGE_API_ORIGIN}

WORKDIR /app

COPY --from=dependencies /app/node_modules ./node_modules
COPY frontend ./

RUN npm run lint
RUN npm run build

FROM node:24-alpine AS runtime

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    HOSTNAME=0.0.0.0 \
    PORT=3000 \
    MLFORGE_API_ORIGIN=http://api:8000

WORKDIR /app

RUN addgroup -S -g 10001 nodejs \
    && adduser -S -u 10001 -G nodejs nextjs

COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["node", "-e", "fetch('http://127.0.0.1:3000/api/health/ready').then((response) => { if (!response.ok) process.exit(1); }).catch(() => process.exit(1));"]

CMD ["node", "server.js"]
