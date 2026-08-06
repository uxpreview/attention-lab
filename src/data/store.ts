import type { Recording, Study } from "./types";

/**
 * IndexedDB-backed persistence. Studies carry a stimulus image blob, and
 * recordings can run to thousands of points, so localStorage is not an option.
 *
 * Everything stays on the participant's machine — no video, no gaze data, and
 * no images leave the browser. That is a hard requirement for running this on
 * real users without a consent and data-handling process.
 */

const DB_NAME = "eyetrack";
const DB_VERSION = 1;
const STUDIES = "studies";
const RECORDINGS = "recordings";

let dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;

  dbPromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STUDIES)) {
        db.createObjectStore(STUDIES, { keyPath: "id" });
      }
      if (!db.objectStoreNames.contains(RECORDINGS)) {
        const store = db.createObjectStore(RECORDINGS, { keyPath: "id" });
        store.createIndex("studyId", "studyId", { unique: false });
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });

  return dbPromise;
}

function tx<T>(
  storeName: string,
  mode: IDBTransactionMode,
  run: (store: IDBObjectStore) => IDBRequest<T>
): Promise<T> {
  return openDb().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const transaction = db.transaction(storeName, mode);
        const request = run(transaction.objectStore(storeName));
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      })
  );
}

export function saveStudy(study: Study): Promise<IDBValidKey> {
  return tx(STUDIES, "readwrite", (store) => store.put(study));
}

export function getStudy(id: string): Promise<Study | undefined> {
  return tx<Study | undefined>(STUDIES, "readonly", (store) => store.get(id));
}

export async function listStudies(): Promise<Study[]> {
  const studies = await tx<Study[]>(STUDIES, "readonly", (store) => store.getAll());
  return studies.sort((a, b) => b.createdAt - a.createdAt);
}

export async function deleteStudy(id: string): Promise<void> {
  const recordings = await listRecordings(id);
  await Promise.all(recordings.map((r) => deleteRecording(r.id)));
  await tx(STUDIES, "readwrite", (store) => store.delete(id));
}

export function saveRecording(recording: Recording): Promise<IDBValidKey> {
  return tx(RECORDINGS, "readwrite", (store) => store.put(recording));
}

export async function listRecordings(studyId: string): Promise<Recording[]> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const store = db.transaction(RECORDINGS, "readonly").objectStore(RECORDINGS);
    const request = store.index("studyId").getAll(studyId);
    request.onsuccess = () =>
      resolve((request.result as Recording[]).sort((a, b) => a.createdAt - b.createdAt));
    request.onerror = () => reject(request.error);
  });
}

export function deleteRecording(id: string): Promise<void> {
  return tx(RECORDINGS, "readwrite", (store) => store.delete(id)).then(() => undefined);
}

export function newId(prefix: string): string {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}
