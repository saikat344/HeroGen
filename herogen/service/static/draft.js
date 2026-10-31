// IndexedDB retains an uploaded Blob across the email sign-in round trip.
const DATABASE = 'herogen-drafts';
export async function draftStore(mode, value) {
  const db = await new Promise((resolve, reject) => {
    const r = indexedDB.open(DATABASE, 1);
    r.onupgradeneeded = () => r.result.createObjectStore('drafts');
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.onblocked = () => reject(new Error('Draft storage is blocked.'));
  });
  try {
    return await new Promise((resolve, reject) => {
      const tx = db.transaction('drafts', mode === 'get' ? 'readonly' : 'readwrite');
      const store = tx.objectStore('drafts');
      const r = mode === 'get' ? store.get('current') : mode === 'put' ? store.put(value, 'current') : store.delete('current');
      let result;
      r.onsuccess = () => { result = r.result; };
      tx.oncomplete = () => resolve(result);
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error || new Error('Draft storage interrupted.'));
    });
  } finally { db.close(); }
}
