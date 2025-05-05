from app.supabase_client import supabase

TABLE = "profiles"

def upsert_profile(user_id: str, profile: dict):
    """
    Inserta o actualiza un perfil de usuario en Supabase.
    user_id puede ser un UUID generado en el frontend o fijo.
    """
    data = {
        "id": user_id,
        "genero": profile["genero"],
        "edad": profile["edad"],
        "estatura": profile["estatura"],
        "peso": profile["peso"],
        "lesion": profile["lesion"],
        "lesion_descripcion": profile["lesion_descripcion"],
    }
    resp = (
        supabase
        .table(TABLE)
        .upsert(data, on_conflict="id")
        .execute()
    )
    return resp

def fetch_profile(user_id: str) -> dict | None:
    """
    Recupera el perfil del usuario con ese user_id.
    Devuelve un dict con las columnas de profiles, o None si no existe o hay error.
    """
    try:
        resp = (
            supabase
            .table(TABLE)
            .select("*")
            .eq("id", user_id)
            .maybe_single()       # devuelve un solo registro o None
            .execute()
        )
    except Exception as e:
        # Algo ha fallado en la llamada HTTP
        print("Error fetching profile:", e)
        return None

    # Si no vino respuesta o no tiene el formato esperado
    if resp is None:
        return None

    # Algunos versions de supabase-py devuelven (data, count, error) como tuple
    # otros un objeto con atributos .data y .error.
    # Lo cubrimos así:
    data = None
    error = None

    if isinstance(resp, tuple) and len(resp) == 3:
        data, count, error = resp
    else:
        data  = getattr(resp, "data", None)
        error = getattr(resp, "error", None)

    if error:
        # log opcional: print("Supabase error:", error)
        return None

    # data puede ser None si no existe el registro
    return data  # si es None, significa “no encontrado”