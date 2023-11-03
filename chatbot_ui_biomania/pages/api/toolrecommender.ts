export const config = {
  runtime: 'edge',
}

const handler = async (req: Request): Promise<Response> => {
  return new Response(JSON.stringify([
    { tool_name: 'Tool 1', tool_desc: 'Description 1' },
    { tool_name: 'Tool 2', tool_desc: 'Description 2' },
    { tool_name: 'Tool 3', tool_desc: 'Description 3' },
    { tool_name: 'Tool 4', tool_desc: 'Description 4' },
    { tool_name: 'Tool 5', tool_desc: 'Description 5' },
    { tool_name: 'Tool 6', tool_desc: 'Description 6' },
    { tool_name: 'Tool 7', tool_desc: 'Description 7' },
    { tool_name: 'Tool 8', tool_desc: 'Description 8' },
    { tool_name: 'Tool 9', tool_desc: 'Description 9' },
    { tool_name: 'Tool 10', tool_desc: 'Description 10' },
    { tool_name: 'Tool 11', tool_desc: 'Description 11' },
    { tool_name: 'Tool 12', tool_desc: 'Description 12' },
  ]), { status: 200 });
}