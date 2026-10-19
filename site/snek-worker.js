// The engine compiled to WebAssembly (snek.js + snek.wasm, built by web/build.sh), off the page's thread.
// Same protocol as `snek serve`: the page posts {id, cmd} and gets back {id, out} with the JSON answer.
// {id, load: name} fetches proof/<name> (the stored proof that red wins) into the engine's file system.
importScripts('snek.js');
const mod = SnekEngine();
const engine = mod.then((m) => m.cwrap('snek_cmd', 'string', ['string']));
onmessage = async (e) => {
  if (e.data.load !== undefined) {
    const m = await mod;
    let ok = false;
    try {
      const r = await fetch('proof/' + e.data.load);
      if (r.ok) {
        try { m.FS.mkdir('/proof'); } catch (x) {}
        m.FS.writeFile('/proof/' + e.data.load, new Uint8Array(await r.arrayBuffer()));
        ok = true;
      }
    } catch (x) {}
    postMessage({ id: e.data.id, out: JSON.stringify({ ok }) });
    return;
  }
  const run = await engine;
  postMessage({ id: e.data.id, out: run(e.data.cmd) });
};
