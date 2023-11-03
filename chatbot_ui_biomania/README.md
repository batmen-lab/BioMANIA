# BioMANIA UI

BioMANIA UI is an open source chat UI for [BioMANIA](https://github.com/batmen-lab/BioMANIA)


## Deploy

**Vercel**

Host your own live version of ToolLLaMA UI with Vercel.

[![Deploy with Vercel](https://vercel.com/button)](https://github.com/batmen-lab/BioMANIA)

**Docker**

Build locally:

```shell
docker build -t chatgpt-ui-biomania .
docker run -e -p 3000:3000 chatgpt-ui-biomania
```

## Running Locally

**1. Clone Repo**

```bash
git clone https://github.com/batmen-lab/BioMANIA.git
```

**2. Install Dependencies**

```bash
npm i
```

**3. Run App**

```bash
npm run dev
```

**5. Use It**

You should be able to start chatting.
