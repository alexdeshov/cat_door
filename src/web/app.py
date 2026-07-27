import shutil
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.common.database import Event, create_session_factory, session_dependency
from src.common.settings import Settings
from src.recorder.ffmpeg import RecorderError, mjpeg_stream, record_clip

LABELS = {"pending", "known_cat", "foreign_cat", "not_visible", "multiple_cats"}


class CapturePayload(BaseModel):
    duration_seconds: int = Field(ge=3, le=30)


class LabelPayload(BaseModel):
    label: str
    cat_name: str | None = Field(default=None, max_length=100)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    factory = create_session_factory(settings.resolved_database_path)
    templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

    app = FastAPI(title="CatDoor AI", version="0.1.0")
    app.state.settings = settings
    app.state.session_factory = factory

    def get_session() -> Generator[Session, None, None]:
        yield from session_dependency(factory)

    @app.get("/", response_class=HTMLResponse)
    def index(
        request: Request,
        session: Annotated[Session, Depends(get_session)],
    ) -> HTMLResponse:
        events = session.scalars(select(Event).order_by(Event.created_at.desc()).limit(100)).all()
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "events": events,
                "face_configured": bool(settings.camera_face_rtsp),
                "side_configured": bool(settings.camera_side_rtsp),
                "default_duration": settings.capture_duration_seconds,
            },
        )

    @app.get("/stream/{view}.mjpeg")
    def stream(view: str) -> StreamingResponse:
        urls = {"face": settings.camera_face_rtsp, "side": settings.camera_side_rtsp}
        if view not in urls:
            raise HTTPException(404, "Неизвестная камера")
        if not urls[view]:
            raise HTTPException(503, "Камера не настроена")
        return StreamingResponse(
            mjpeg_stream(settings.ffmpeg_path, urls[view], settings.preview_fps),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.post("/api/events", status_code=201)
    def capture(
        payload: CapturePayload,
        session: Annotated[Session, Depends(get_session)],
    ) -> dict[str, object]:
        if not settings.camera_face_rtsp:
            raise HTTPException(503, "Камера face не настроена")
        event = Event(duration_seconds=payload.duration_seconds)
        session.add(event)
        session.commit()
        event_dir = settings.data_dir / "events" / f"{event.id:08d}"
        recordings = {"face": (settings.camera_face_rtsp, event_dir / "face.mp4")}
        if settings.camera_side_rtsp:
            recordings["side"] = (settings.camera_side_rtsp, event_dir / "side.mp4")
        errors: list[str] = []
        completed: set[str] = set()
        with ThreadPoolExecutor(max_workers=len(recordings)) as executor:
            futures = {
                executor.submit(
                    record_clip,
                    settings.ffmpeg_path,
                    url,
                    destination,
                    payload.duration_seconds,
                ): (view, destination)
                for view, (url, destination) in recordings.items()
            }
            for future in as_completed(futures):
                view, destination = futures[future]
                try:
                    future.result()
                    completed.add(view)
                    relative = destination.relative_to(settings.data_dir).as_posix()
                    if view == "face":
                        event.media_path = relative
                    else:
                        event.side_media_path = relative
                except RecorderError as exc:
                    errors.append(f"{view}:{exc}")
        event.status = "ready" if not errors else ("partial" if completed else "failed")
        event.error_code = ",".join(errors) or None
        session.commit()
        return {"id": event.id, "status": event.status}

    @app.post("/api/events/{event_id}/label")
    def label_event(
        event_id: int,
        payload: LabelPayload,
        session: Annotated[Session, Depends(get_session)],
    ) -> dict[str, object]:
        event = session.get(Event, event_id)
        if not event:
            raise HTTPException(404, "Событие не найдено")
        if payload.label not in LABELS:
            raise HTTPException(422, "Неизвестный тип разметки")
        cat_name = payload.cat_name.strip() if payload.cat_name else None
        if payload.label == "known_cat" and not cat_name:
            raise HTTPException(422, "Для домашнего кота укажите имя")
        if payload.label != "known_cat" and cat_name:
            raise HTTPException(422, "Имя допустимо только для домашнего кота")
        event.label = payload.label
        event.cat_name = cat_name
        session.commit()
        return {"id": event.id, "label": event.label, "cat_name": event.cat_name}

    @app.delete("/api/events/{event_id}", status_code=204)
    def delete_event(
        event_id: int,
        session: Annotated[Session, Depends(get_session)],
    ) -> None:
        event = session.get(Event, event_id)
        if not event:
            raise HTTPException(404, "Событие не найдено")

        events_root = (settings.data_dir / "events").resolve()
        event_dir = (events_root / f"{event.id:08d}").resolve()
        if events_root not in event_dir.parents:
            raise HTTPException(500, "Небезопасный путь события")
        if event_dir.exists():
            try:
                shutil.rmtree(event_dir)
            except OSError as exc:
                raise HTTPException(500, "Не удалось удалить файлы события") from exc

        session.delete(event)
        session.commit()

    @app.get("/media/{event_id}/{view}")
    def media(
        event_id: int,
        view: str,
        session: Annotated[Session, Depends(get_session)],
    ) -> FileResponse:
        event = session.get(Event, event_id)
        if view not in {"face", "side"}:
            raise HTTPException(404, "Неизвестная камера")
        media_path = (
            event.media_path if event and view == "face" else event.side_media_path if event else None
        )
        if not event or not media_path:
            raise HTTPException(404, "Видео не найдено")
        root = settings.data_dir.resolve()
        path = (root / media_path).resolve()
        if root not in path.parents or not path.is_file():
            raise HTTPException(404, "Видео не найдено")
        return FileResponse(
            path,
            media_type="video/mp4",
            filename=f"event-{event.id}-{view}.mp4",
        )

    return app


def run() -> None:
    settings = Settings()
    uvicorn.run(create_app(settings), host=settings.web_host, port=settings.web_port)


if __name__ == "__main__":
    run()
