"""Return domain lookup failures to agents as evidence they can recover from."""
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

class LookupErrorMiddleware(AgentMiddleware):
    def _error(self,request,exc):
        # ValueError messages are our domain lookup hints, not provider credentials.
        text=str(exc) if isinstance(exc,ValueError) else 'The source is temporarily unavailable.'
        return ToolMessage(content='Lookup failed: '+text+' Do not invent a record. Ask for a valid ID or explain missing evidence.',tool_call_id=request.tool_call['id'],status='error')

    def wrap_tool_call(self,request,handler):
        try:return handler(request)
        except (ValueError,RuntimeError) as exc:return self._error(request,exc)

    async def awrap_tool_call(self,request,handler):
        try:return await handler(request)
        except (ValueError,RuntimeError) as exc:return self._error(request,exc)
