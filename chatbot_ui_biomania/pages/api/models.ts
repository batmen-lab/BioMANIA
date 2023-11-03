export const config = {
  runtime: 'edge',
};

const handler = async (req: Request): Promise<Response> => {
  return new Response(JSON.stringify([
    {id: 'single-tool', name: 'Single Tool'},
  ]), { status: 200 });
};

export default handler;
