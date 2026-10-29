"""Return domain lookup failures to agents as evidence they can recover from."""
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

def lookup_error(exc,tool_call_id):
    """Build the recoverable error result for a failed lookup."""
    # ValueError messages are our domain lookup hints, not provider credentials.
    text=str(exc) if isinstance(exc,ValueError) else 'The source is temporarily unavailable.'
    return ToolMessage(content='Lookup failed: '+text+' Do not invent a record. Ask for a valid ID or explain missing evidence.',tool_call_id=tool_call_id,status='error')

class LookupErrorMiddleware(AgentMiddleware):
    def _error(self,request,exc):
        return lookup_error(exc,request.tool_call['id'])

    def wrap_tool_call(self,request,handler):
        try:return handler(request)
        except (ValueError,RuntimeError) as exc:return self._error(request,exc)

    async def awrap_tool_call(self,request,handler):
        try:return await handler(request)
        except (ValueError,RuntimeError) as exc:return self._error(request,exc)
