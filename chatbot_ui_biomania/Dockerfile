
FROM node:19-alpine as build
WORKDIR /app
COPY package*.json ./

RUN apk --no-cache add curl

FROM dependencies AS build
COPY . .
RUN npm run build

FROM node:19-alpine AS production
WORKDIR /app
COPY --from=dependencies /app/node_modules ./node_modules
COPY --from=build /app/.next ./.next
COPY --from=build /app/public ./public
COPY --from=build /app/package*.json ./
COPY --from=build /app/next.config.js ./next.config.js
COPY --from=build /app/next-i18next.config.js ./next-i18next.config.js

EXPOSE 3000

CMD ["npm", "run", "dev"]
