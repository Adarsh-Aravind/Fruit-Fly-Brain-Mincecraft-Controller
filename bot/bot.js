// Mineflayer "body" for the MaleCNS fly brain.
//
// Each bot streams fly-like senses (compound-eye luminance, looming, odours, taste, touch,
// wind) to the Python brain over a WebSocket and applies the motor commands it gets back.
//
//   node bot.js --count 1 --viewer 3007
//
const mineflayer = require('mineflayer')
const { Vec3 } = require('vec3')
const WebSocket = require('ws')

const args = parseArgs(process.argv.slice(2), {
  host: '127.0.0.1', port: 25565, version: '1.21.4', count: 1, prefix: 'Fly',
  brain: 'ws://127.0.0.1:8765', viewer: 3007, hz: 10
})

// Must match brain/flybrain/senses.py and motor.py
// Fly-like panoramic vision: each eye spans 180 deg (15 deg binocular overlap in front,
// 30 deg blind spot behind), 140 deg vertically. Minecraft's own max FOV is only 110 deg.
const EYE_ROWS = 12
const EYE_COLS = 16
const EYE_AZIMUTH = [-15, 165]
const EYE_ELEVATION = 70
const EYE_RANGE = 24
const TURN_MAX_DEG = 45
const PITCH_LIMIT = 60 * Math.PI / 180

const HOSTILE = new Set(['zombie', 'skeleton', 'creeper', 'spider', 'cave_spider', 'witch', 'enderman', 'drowned',
  'husk', 'stray', 'slime', 'phantom', 'pillager', 'vindicator', 'silverfish', 'zombie_villager', 'bogged', 'breeze'])
const ANIMALS = new Set(['cow', 'pig', 'chicken', 'sheep', 'rabbit', 'mooshroom', 'goat'])
const FOOD_BLOCKS = ['sweet_berry_bush', 'melon', 'pumpkin', 'carrots', 'potatoes', 'wheat', 'cake']
const DANGER_BLOCKS = ['lava', 'fire', 'soul_fire', 'cactus', 'magma_block', 'sweet_berry_bush', 'campfire']
const PLANT_BLOCKS_RX = /(_log|_leaves|_wood)$/

for (let i = 0; i < args.count; i++) setTimeout(() => startBody(i), i * 1500)

function startBody (id) {
  const username = `${args.prefix}${id}`
  const bot = mineflayer.createBot({ host: args.host, port: args.port, username, version: args.version, auth: 'offline' })
  const mcData = require('minecraft-data')(args.version)
  const foodNames = new Set(mcData.foodsArray.map(f => f.name))
  const idsOf = names => names.map(n => mcData.blocksByName[n]?.id).filter(x => x !== undefined)
  const foodBlockIds = idsOf(FOOD_BLOCKS)
  const dangerBlockIds = idsOf(DANGER_BLOCKS)

  const st = {
    ws: null, ready: false, prevLum: null, prevAngular: new Map(), prevHealth: 20, statHealth: 20, prevFood: 20,
    blockCache: { t: 0, food: [], danger: [], plants: [] }, noise: 0, digging: false, lastAttack: 0,
    stats: { mined: 0, collected: 0, eaten: 0, damage: 0, deaths: 0, logs: 0 }, action: null, lastActionAt: 0,
    // speedrun bookkeeping (reset by the brain at the start of every attempt)
    episodeStart: Date.now(), progress: {}, logsThisEpisode: 0, busy: false, lastCraft: '',
    // lifecycle handles, so a reconnect never leaves a second stream feeding the same brain slot
    senseTimer: null, reconnectTimer: null, ended: false, lastDeath: 0,
    stoneCache: { t: 0, dist: null }
  }

  bot.once('spawn', () => {
    console.log(`[${username}] spawned`)
    st.ready = true
    if (id === 0 && args.viewer) {
      try {
        require('prismarine-viewer').mineflayer(bot, { port: +args.viewer, firstPerson: true })
        console.log(`[${username}] viewer on http://localhost:${args.viewer}`)
      } catch (e) { console.log('viewer failed:', e.message) }
    }
    connectBrain()
    st.senseTimer = setInterval(senseAndSend, 1000 / args.hz)
  })
  bot.on('death', () => {
    // mineflayer can repeat this event while the death screen is up: count real deaths once
    const now = Date.now()
    if (now - st.lastDeath < 3000) return
    st.lastDeath = now
    st.stats.deaths++
    st.prevLum = null
  })
  bot.on('health', () => {
    const lost = st.statHealth - bot.health
    if (lost > 0) st.stats.damage += lost
    st.statHealth = bot.health
    if (bot.food > st.prevFood) st.stats.eaten += bot.food - st.prevFood
    st.prevFood = bot.food
  })
  bot.on('playerCollect', (collector, item) => {
    if (collector !== bot.entity) return
    st.stats.collected++
    const meta = item.getDroppedItem?.()
    if (meta && /_log$/.test(meta.name)) { st.stats.logs += meta.count; st.logsThisEpisode += meta.count }
  })
  bot.on('soundEffectHeard', (name, pos) => {
    if (pos && bot.entity && pos.distanceTo(bot.entity.position) < 16) st.noise = 1
  })
  bot.on('kicked', r => console.log(`[${username}] kicked:`, r))
  bot.on('error', e => console.log(`[${username}] error:`, e.message))
  bot.on('end', () => {
    if (st.ended) return
    st.ended = true
    clearInterval(st.senseTimer)
    clearTimeout(st.reconnectTimer)
    if (st.ws) { const old = st.ws; st.ws = null; old.removeAllListeners('close'); old.close() }
    console.log(`[${username}] disconnected, reconnecting in 5 s`)
    setTimeout(() => startBody(id), 5000)
  })

  function connectBrain () {
    if (st.ended) return
    const ws = new WebSocket(args.brain)
    ws.on('open', () => {
      if (st.ended) { ws.close(); return }
      st.ws = ws
      ws.send(JSON.stringify({ type: 'hello', id, name: username }))
    })
    ws.on('message', data => {
      const msg = JSON.parse(data)
      if (msg.type === 'action') { st.action = msg.action; st.lastActionAt = Date.now(); act(msg.action) }
      if (msg.type === 'chat') bot.chat(msg.text)
      if (msg.type === 'reset') resetEpisode()
    })
    ws.on('close', () => {
      if (st.ws === ws) st.ws = null
      if (st.ended) return
      release()
      st.reconnectTimer = setTimeout(connectBrain, 2000)
    })
    ws.on('error', () => {})
  }

  function resetEpisode () {
    st.episodeStart = Date.now()
    st.progress = {}
    st.logsThisEpisode = 0
    st.busy = false
    st.lastCraft = ''
    release()
  }

  function release () {
    bot.clearControlStates()
    if (st.digging) { bot.stopDigging(); st.digging = false }
  }

  // ---------------------------------------------------------------- senses
  function senseAndSend () {
    if (st.ended || !st.ready || !st.ws || st.ws.readyState !== WebSocket.OPEN || !bot.entity) return
    if (Date.now() - st.lastActionAt > 2000) release() // brain went quiet: go limp
    const obs = sense()
    st.ws.send(JSON.stringify({ type: 'obs', id, obs }))
  }

  function sense () {
    const e = bot.entity
    const dead = bot.health <= 0
    const eyePos = e.position.offset(0, e.eyeHeight ?? 1.62, 0)
    const now = Date.now()
    if (now - st.blockCache.t > 1000) refreshBlockCache()

    const daylight = skyBrightness()
    const living = Object.values(bot.entities).filter(o => o !== e && o.position && (o.type === 'mob' || o.type === 'player' || o.type === 'hostile' || o.type === 'animal') && o.position.distanceTo(e.position) < EYE_RANGE)
    const items = Object.values(bot.entities).filter(o => o.name === 'item' && o.position.distanceTo(e.position) < 16)

    // Compound eyes: left eye looks from 15 deg right of heading round to 165 deg left, mirrored for the right eye.
    const eye = { L: [], R: [] }
    for (const side of ['L', 'R']) {
      for (let r = 0; r < EYE_ROWS; r++) {
        const pitch = e.pitch + deg(EYE_ELEVATION - 2 * EYE_ELEVATION * r / (EYE_ROWS - 1))
        for (let c = 0; c < EYE_COLS; c++) {
          const off = deg(EYE_AZIMUTH[0] + (EYE_AZIMUTH[1] - EYE_AZIMUTH[0]) * c / (EYE_COLS - 1))
          const yaw = e.yaw + (side === 'L' ? off : -off)
          eye[side].push(rayLuminance(eyePos, yaw, pitch, daylight, living))
        }
      }
    }
    const eyeDelta = { L: eye.L.map((v, i) => st.prevLum ? v - st.prevLum.L[i] : 0), R: eye.R.map((v, i) => st.prevLum ? v - st.prevLum.R[i] : 0) }
    st.prevLum = eye

    // Looming: rate of increase of an object's angular size, per side.
    const loom = { L: 0, R: 0 }
    const seen = new Map()
    for (const o of living) {
      const d = Math.max(0.3, o.position.distanceTo(e.position))
      const ang = 2 * Math.atan((o.height || 1) / 2 / d)
      const prev = st.prevAngular.get(o.id)
      seen.set(o.id, ang)
      if (prev === undefined) continue
      const growth = (ang - prev) * args.hz // rad/s
      if (growth <= 0) continue
      const weight = HOSTILE.has(o.name) ? 1 : (o.type === 'player' ? 0.7 : 0.4)
      const b = bearing(o.position)
      loom[b >= 0 ? 'L' : 'R'] += weight * growth / 0.8
    }
    st.prevAngular = seen

    // Odours: exponential falloff with distance, antenna asymmetry by bearing.
    const food = { L: 0, R: 0 }; const danger = { L: 0, R: 0 }; const plant = { L: 0, R: 0 }
    const smell = (acc, pos, w, scale) => {
      const d = pos.distanceTo(e.position)
      const s = w * Math.exp(-d / scale)
      const b = Math.sin(bearing(pos))
      acc.L += s * (0.5 + 0.5 * b); acc.R += s * (0.5 - 0.5 * b)
    }
    for (const o of living) {
      if (ANIMALS.has(o.name)) smell(food, o.position, 0.6, 8)
      if (HOSTILE.has(o.name)) smell(danger, o.position, 1.0, 6)
    }
    for (const it of items) smell(isFoodItem(it) ? food : plant, it.position, 0.8, 6)
    for (const p of st.blockCache.food) smell(food, p, 0.5, 6)
    for (const p of st.blockCache.danger) smell(danger, p, 0.8, 3)
    for (const p of st.blockCache.plants) smell(plant, p, 0.15, 8)

    // Taste: contact chemistry.
    const within = (pos, r) => pos.distanceTo(e.position) < r
    let sugar = 0; let bitter = 0
    if (items.some(it => isFoodItem(it) && within(it.position, 1.8))) sugar = 1
    if (st.blockCache.food.some(p => within(p, 1.8))) sugar = Math.max(sugar, 0.8)
    if (bot.usingHeldItem && bot.heldItem && foodNames.has(bot.heldItem.name)) sugar = 1
    if (living.some(o => HOSTILE.has(o.name) && within(o.position, 2.5))) bitter = 1
    if (st.blockCache.danger.some(p => within(p, 2))) bitter = Math.max(bitter, 0.8)

    // Touch: short horizontal feelers left/right, plus head-on collision.
    const feel = (off) => {
      const blk = bot.world.raycast(e.position.offset(0, 0.9, 0), dirVec(e.yaw + off, 0), 0.7)
      return blk ? 0.6 : 0
    }
    const blocked = e.isCollidedHorizontally ? 0.7 : 0
    const touch = { L: Math.max(blocked, feel(deg(60))), R: Math.max(blocked, feel(deg(-60))) }

    const v = e.velocity
    const wind = Math.min(1, Math.max(0, -v.y - 0.1) * 1.5 + Math.hypot(v.x, v.z) * (e.onGround ? 0 : 1.2))
    const lost = Math.max(0, st.prevHealth - bot.health)
    st.prevHealth = bot.health
    const noise = st.noise; st.noise *= 0.5

    return {
      dead, eye, eye_delta: eyeDelta, loom, food_odor: food, danger_odor: danger, plant_odor: plant,
      taste_sugar: sugar, taste_bitter: bitter, touch, wind, hurt: Math.min(1, lost / 4), noise,
      health: bot.health, food: bot.food, pos: [e.position.x, e.position.y, e.position.z], yaw: e.yaw, pitch: e.pitch,
      on_ground: e.onGround, in_water: !!e.isInWater, day: daylight, stats: { ...st.stats },
      inventory: inventorySummary(), target: cursorTarget(), ...speedrunState(),
      nearest_log_dist: nearestDist(st.blockCache.plants), nearest_stone_dist: nearestStone()
    }
  }

  function rayLuminance (from, yaw, pitch, daylight, living) {
    const dir = dirVec(yaw, pitch)
    const blk = bot.world.raycast(from, dir, EYE_RANGE)
    const dBlock = blk ? from.distanceTo(blk.intersect ?? blk.position) : EYE_RANGE
    // Entities are dark silhouettes in front of the scene.
    for (const o of living) {
      const to = o.position.offset(0, (o.height || 1) / 2, 0).minus(from)
      const d = to.norm()
      if (d > dBlock || d < 0.2) continue
      const cos = to.dot(dir) / d
      if (cos > Math.cos(Math.atan(Math.max(o.width || 0.6, o.height || 1) / 2 / d) + 0.05)) return 0.05
    }
    if (!blk) return pitch > -0.1 ? daylight : 0.3 * daylight
    const above = blk.position.offset(0, 1, 0)
    const sky = (bot.world.getSkyLight(above) ?? 15) / 15 * daylight
    const lamp = (bot.world.getBlockLight(above) ?? 0) / 15
    const light = Math.max(sky, lamp)
    const shade = /water|leaves/.test(blk.name) ? 0.6 : 1
    return Math.max(0, Math.min(1, light * shade * (0.55 + 0.45 * (1 - dBlock / EYE_RANGE))))
  }

  function skyBrightness () {
    const t = bot.time?.timeOfDay ?? 6000
    if (t < 12000) return 1
    if (t < 13500) return 1 - 0.85 * (t - 12000) / 1500
    if (t < 22500) return 0.15
    return 0.15 + 0.85 * (t - 22500) / 1500
  }

  function refreshBlockCache () {
    const e = bot.entity
    st.blockCache.t = Date.now()
    st.blockCache.food = foodBlockIds.length ? bot.findBlocks({ matching: foodBlockIds, maxDistance: 16, count: 20 }) : []
    st.blockCache.danger = dangerBlockIds.length ? bot.findBlocks({ matching: dangerBlockIds, maxDistance: 10, count: 20 }) : []
    st.blockCache.plants = bot.findBlocks({ matching: b => PLANT_BLOCKS_RX.test(b.name) && b.name.endsWith('_log'), maxDistance: 16, count: 20 })
    const c = e.position
    for (const k of ['food', 'danger', 'plants']) st.blockCache[k] = st.blockCache[k].map(p => p.offset(0.5, 0.5, 0.5)).filter(p => p.distanceTo(c) < 17)
  }

  function nearestDist (points) {
    if (!points.length) return null
    const p = bot.entity.position
    return Math.min(...points.map(q => q.distanceTo(p)))
  }

  function nearestStone () {
    const now = Date.now()
    if (now - st.stoneCache.t > 1000) {
      st.stoneCache.t = now
      const blk = bot.findBlock({ matching: b => b.name === 'stone' || b.name === 'cobblestone', maxDistance: 16 })
      st.stoneCache.pos = blk ? blk.position.offset(0.5, 0.5, 0.5) : null
    }
    return st.stoneCache.pos ? st.stoneCache.pos.distanceTo(bot.entity.position) : null
  }

  function isFoodItem (it) {
    const meta = it.getDroppedItem?.()
    return !!meta && foodNames.has(meta.name)
  }

  function bearing (pos) { // + = to the left of heading
    const e = bot.entity
    const dx = pos.x - e.position.x; const dz = pos.z - e.position.z
    const target = Math.atan2(-dx, -dz)
    let d = target - e.yaw
    while (d > Math.PI) d -= 2 * Math.PI
    while (d < -Math.PI) d += 2 * Math.PI
    return d
  }

  function cursorTarget () {
    const e = bot.entity
    const from = e.position.offset(0, e.eyeHeight ?? 1.62, 0)
    const ent = bot.entityAtCursor(3.5)
    if (ent) return { kind: 'entity', name: ent.name ?? ent.username ?? 'entity' }
    const blk = bot.world.raycast(from, dirVec(e.yaw, e.pitch), 4.5)
    if (blk) return { kind: 'block', name: blk.name }
    return null
  }

  function inventorySummary () {
    let logs = 0; let food = 0; let total = 0
    for (const it of bot.inventory.items()) {
      total += it.count
      if (/_log$/.test(it.name)) logs += it.count
      if (foodNames.has(it.name)) food += it.count
    }
    return { logs, food, total }
  }

  // ---------------------------------------------------------------- speedrun
  const KEY_ITEMS = ['log', 'planks', 'stick', 'crafting_table', 'wooden_pickaxe', 'cobblestone', 'stone_pickaxe',
    'raw_iron', 'furnace', 'iron_ingot', 'coal']
  const STONE_RX = /^(cobblestone|cobbled_deepslate|blackstone)$/

  function itemCounts () {
    const c = Object.fromEntries(KEY_ITEMS.map(k => [k, 0]))
    for (const it of bot.inventory.items()) {
      if (/_log$|_stem$/.test(it.name)) c.log += it.count
      else if (/_planks$/.test(it.name)) c.planks += it.count
      else if (STONE_RX.test(it.name)) c.cobblestone += it.count
      else if (it.name in c) c[it.name] += it.count
    }
    return c
  }

  function nearBlock (name) {
    return bot.findBlock({ matching: b => b.name === name, maxDistance: 4 })
  }

  function speedrunState () {
    const items = itemCounts()
    const t = (Date.now() - st.episodeStart) / 1000
    const mark = (name, cond) => { if (cond && st.progress[name] === undefined) st.progress[name] = +t.toFixed(1) }
    const tableNear = !!nearBlock('crafting_table')
    const furnaceNear = !!nearBlock('furnace')
    mark('log', items.log > 0 || st.logsThisEpisode > 0)
    mark('logs3', st.logsThisEpisode >= 3)
    mark('planks', items.planks > 0)
    mark('crafting_table', items.crafting_table > 0 || (tableNear && st.lastCraft === 'crafting_table'))
    mark('wooden_pickaxe', items.wooden_pickaxe > 0)
    mark('cobblestone', items.cobblestone > 0)
    mark('stone_pickaxe', items.stone_pickaxe > 0)
    mark('raw_iron', items.raw_iron > 0)
    mark('furnace', items.furnace > 0 || (furnaceNear && st.lastCraft === 'furnace'))
    mark('iron_ingot', items.iron_ingot > 0)
    const table = tableNear || items.crafting_table > 0
    const furnace = furnaceNear || items.furnace > 0
    // Feasible AND useful for the speedrun: no duplicate stations/tools, no hoarding sticks or planks.
    const craftable = [
      true,
      items.log >= 1 && items.planks < 12,
      items.planks >= 2 && items.stick < 8,
      items.planks >= 4 && !table,
      items.planks >= 3 && items.stick >= 2 && table && items.wooden_pickaxe === 0 && items.stone_pickaxe === 0,
      items.cobblestone >= 3 && items.stick >= 2 && table && items.stone_pickaxe === 0,
      items.cobblestone >= 8 && table && !furnace,
      items.raw_iron >= 1 && furnace && (items.planks + items.log + items.coal) >= 1
    ]
    return {
      items, progress: { ...st.progress }, episode_s: t, busy: st.busy || st.digging,
      near: { crafting_table: tableNear ? 1 : 0, furnace: furnaceNear ? 1 : 0 }, craftable
    }
  }

  const itemId = n => mcData.itemsByName[n]?.id

  async function craftItem (name, count, table) {
    const id = itemId(name)
    const recipe = id !== undefined && bot.recipesFor(id, null, 1, table)[0]
    if (!recipe) throw new Error('no recipe for ' + name)
    await bot.craft(recipe, count, table)
  }

  async function ensureStation (name) {
    let blk = nearBlock(name)
    if (blk) return blk
    const item = bot.inventory.items().find(i => i.name === name)
    if (!item) throw new Error('no ' + name)
    const base = bot.entity.position.floored()
    const spots = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1], [2, 0], [0, 2], [-2, 0], [0, -2]]
    for (const [dx, dz] of spots) {
      const spot = base.offset(dx, 0, dz)
      const below = bot.blockAt(spot.offset(0, -1, 0))
      const here = bot.blockAt(spot)
      if (below && below.boundingBox === 'block' && here && here.boundingBox === 'empty') {
        await bot.equip(item, 'hand')
        await bot.placeBlock(below, new Vec3(0, 1, 0))
        blk = nearBlock(name)
        if (blk) return blk
      }
    }
    throw new Error('nowhere to place ' + name)
  }

  async function doCraft (what) {
    const items = bot.inventory.items()
    if (what === 'planks') {
      const log = items.find(i => /_log$|_stem$/.test(i.name))
      await craftItem(log.name.replace(/_log$|_stem$/, '_planks'), 1, null)
    } else if (what === 'sticks') {
      await craftItem('stick', 1, null)
    } else if (what === 'crafting_table') {
      await craftItem('crafting_table', 1, null)
    } else if (what === 'wooden_pickaxe' || what === 'stone_pickaxe' || what === 'furnace') {
      const table = await ensureStation('crafting_table')
      await craftItem(what, 1, table)
    } else if (what === 'smelt_iron') {
      const blk = await ensureStation('furnace')
      const furnace = await bot.openFurnace(blk)
      try {
        const fuel = items.find(i => i.name === 'coal') || items.find(i => /_planks$/.test(i.name)) || items.find(i => /_log$/.test(i.name))
        const ore = items.find(i => i.name === 'raw_iron')
        if (!furnace.fuelItem() && fuel) await furnace.putFuel(fuel.type, null, Math.min(fuel.count, 2))
        if (ore) await furnace.putInput(ore.type, null, Math.min(ore.count, 3))
        const deadline = Date.now() + 25000
        while (!furnace.outputItem() && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 500))
        if (furnace.outputItem()) await furnace.takeOutput()
      } finally { furnace.close() }
    }
    st.lastCraft = what
  }

  function craft (what) {
    if (!what || what === 'none' || st.busy) return
    st.busy = true
    bot.clearControlStates()
    if (st.digging) { bot.stopDigging(); st.digging = false }
    doCraft(what)
      .catch(e => console.log('[' + username + '] craft ' + what + ' failed: ' + e.message))
      .finally(() => { st.busy = false })
  }

  // ---------------------------------------------------------------- motor
  function act (a) {
    if (!bot.entity || bot.health <= 0) return
    if (a.craft && a.craft !== 'none') craft(a.craft)
    if (st.busy) return // hands are full: crafting / smelting in progress
    // Sticky mining: keep breaking the block unless the brain walks away.
    if (st.digging) {
      if (a.move !== 0 && !a.attack) { bot.stopDigging(); st.digging = false } else return
    }
    bot.setControlState('forward', a.move === 1)
    bot.setControlState('back', a.move === 2)
    bot.setControlState('jump', !!a.jump)
    bot.setControlState('sprint', !!a.sprint && a.move === 1)
    const yaw = bot.entity.yaw + deg(Math.max(-TURN_MAX_DEG, Math.min(TURN_MAX_DEG, a.turn || 0)))
    const pitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, bot.entity.pitch + deg(a.pitch || 0)))
    bot.look(yaw, pitch, true).catch(() => {})

    if (a.use) eat()
    if (a.attack) strike()
  }

  function strike () {
    const now = Date.now()
    const ent = bot.entityAtCursor(3.5)
    if (ent) {
      if (now - st.lastAttack > 550) { bot.attack(ent); st.lastAttack = now }
      return
    }
    if (st.digging) return
    const e = bot.entity
    const blk = bot.world.raycast(e.position.offset(0, e.eyeHeight ?? 1.62, 0), dirVec(e.yaw, e.pitch), 4.5)
    if (!blk || blk.diggable === false || blk.hardness === null || blk.hardness < 0 || blk.name === 'water' || blk.name === 'lava') return
    const real = bot.blockAt(blk.position)
    if (!real || !bot.canDigBlock(real)) return
    st.digging = true
    equipToolFor(real)
      .then(() => bot.dig(real, true))
      .then(() => { st.stats.mined++ })
      .catch(() => {})
      .finally(() => { st.digging = false })
  }

  async function equipToolFor (block) {
    const wantsAxe = /_log$|_planks$|_wood$|_stem$|crafting_table/.test(block.name)
    const tool = bot.inventory.items().find(i => wantsAxe ? /_axe$/.test(i.name) : /_pickaxe$/.test(i.name))
    if (tool && bot.heldItem?.name !== tool.name) await bot.equip(tool, 'hand')
  }

  function eat () {
    if (bot.food >= 20 || bot.usingHeldItem) return
    const food = bot.inventory.items().find(i => foodNames.has(i.name))
    if (!food) return
    bot.equip(food, 'hand').then(() => bot.consume()).catch(() => {})
  }
}

function dirVec (yaw, pitch) {
  return new Vec3(-Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), -Math.cos(yaw) * Math.cos(pitch))
}
function deg (x) { return x * Math.PI / 180 }

function parseArgs (argv, defaults) {
  const out = { ...defaults }
  for (let i = 0; i < argv.length; i++) {
    if (!argv[i].startsWith('--')) continue
    const key = argv[i].slice(2)
    const val = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[++i] : true
    out[key] = typeof defaults[key] === 'number' ? Number(val) : val
  }
  return out
}
