# ---- Base Node ----
FROM node:19-alpine AS base
WORKDIR /app
COPY package*.json ./

# ---- Dependencies ----
FROM base AS dependencies
RUN npm install

# ---- Build ----
FROM dependencies AS build
COPY . .
RUN npm run build

# ---- Development ----
FROM build AS development
CMD ["npm", "run", "dev"]

# ---- Production ----
FROM node:19-alpine AS production
WORKDIR /app


COPY --from=build /app/.next ./.next
COPY --from=build /app/public ./public

# Expose the port the app will run on
EXPOSE 3000

# Start the application
CMD ["npm", "start"]