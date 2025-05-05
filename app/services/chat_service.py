from app.supabase_client import supabase
import uuid

TABLE = "chat_messages"

def save_message(user_id: str, role: str, content: str):
    """
    Guarda un mensaje en la tabla chat_messages.
    """
    data = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "role": role,
        "content": content,
        # Supabase llenará created_at automáticamente si configuras la columna con DEFAULT now()
    }
    supabase.table(TABLE).insert(data).execute()

def fetch_history(user_id: str):
    """
    Recupera el historial de chat de un usuario, ordenado por creación.
    """
    resp = (
        supabase
        .table(TABLE)
        .select("role, content")
        .eq("user_id", user_id)
        .order("created_at", ascending=True)
        .execute()
    )
    return resp.data  # lista de dicts {"role":..., "content":...}