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
        // Resolve on the transaction completing, not the request succeeding:
        // a write request "succeeds" before it is committed, and a tab closed
        // in that window would have reported a save that never happened.
        transaction.oncomplete = () => resolve(request.result);
        transaction.onerror = () => reject(transaction.error);
        transaction.onabort = () => reject(transaction.error);
      })
  );
}

export function saveStudy(study: Study): Promise<IDBValidKey> {
  return tx(STUDIES, "readwrite", (store) => store.put(study));
}

export async function listStudies(): Promise<Study[]> {
  const studies = await tx<Study[]>(STUDIES, "readonly", (store) => store.getAll());
  return studies.sort((a, b) => b.createdAt - a.createdAt);
}

export async function deleteStudy(id: string): Promise<void> {
  // One transaction over both stores, so the study and its recordings go
  // together or not at all — a crash mid-delete must not leave orphaned
  // recordings behind, or a study whose data is half gone.
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction([STUDIES, RECORDINGS], "readwrite");
    const cursorRequest = transaction
      .objectStore(RECORDINGS)
      .index("studyId")
      .openCursor(id);
    cursorRequest.onsuccess = () => {
      const cursor = cursorRequest.result;
      if (cursor) {
        cursor.delete();
        cursor.continue();
      }
    };
    transaction.objectStore(STUDIES).delete(id);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error);
  });
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
