# taskos_sim.py
import math, random
from dataclasses import dataclass
from typing import List, Dict
import simpy, pandas as pd

@dataclass
class TaskType:
    name: str
    mean_duration: float
    skill: str
    base_payout: float
    sla_minutes: float

@dataclass
class PropertyNode:
    id: int
    name: str
    lat: float
    lon: float
    task_rates_per_hour: Dict[str, float]

@dataclass
class Worker:
    id: int
    name: str
    lat: float
    lon: float
    skills: List[str]
    speed_kmph: float = 20.0
    acceptance_rate: float = 0.9
    reliability: float = 0.98

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def travel_minutes_km(distance_km, speed_kmph):
    return (distance_km / max(speed_kmph, 1e-6)) * 60.0

class TaskOSSim:
    def __init__(self, env, properties, workers, task_types, max_radius_km=15.0):
        self.env = env
        self.properties = properties
        self.workers = workers
        self.task_types = task_types
        self.max_radius_km = max_radius_km
        self.event_log = []
        for p in properties:
            env.process(self.property_task_generator(p))

    def log(self, rec):
        self.event_log.append(rec)

    def property_task_generator(self, prop):
        while True:
            yield self.env.timeout(1)
            for tname, rate_per_hour in prop.task_rates_per_hour.items():
                lam_per_min = rate_per_hour / 60.0
                arrivals = 1 if random.random() < lam_per_min else 0
                for _ in range(arrivals):
                    self.env.process(self.dispatch_task(prop, tname))

    def find_candidates(self, task_type, lat, lon):
        c = []
        for w in self.workers:
            if task_type.skill not in w.skills:
                continue
            d = haversine_km(lat, lon, w.lat, w.lon)
            if d <= self.max_radius_km:
                c.append((w, d))
        c.sort(key=lambda x: x[1])
        return c

    def sample_duration(self, mean_minutes):
        sigma = 0.5
        mu = math.log(mean_minutes) - 0.5 * sigma**2
        return max(1.0, random.lognormvariate(mu, sigma))

    def offer_to_worker(self, worker, task_id):
        return random.random() < worker.acceptance_rate

    def complete_task(self, worker, task_id):
        return random.random() < worker.reliability

    def pay(self, base_payout, wait_minutes, sla_minutes):
        return base_payout * 1.1 if wait_minutes <= sla_minutes else base_payout * 0.9

    def dispatch_task(self, prop, task_type_name):
        start_time = self.env.now
        ttype = self.task_types[task_type_name]
        task_id = f"T{int(self.env.now)}-{prop.id}-{random.randint(1000,9999)}"
        candidates = self.find_candidates(ttype, prop.lat, prop.lon)
        if not candidates:
            self.log({"time": self.env.now, "event": "task_failed_no_candidates", "task_id": task_id, "property": prop.name, "type": task_type_name})
            return
        accepted_by, dist_km = None, None
        for worker, d in candidates[:10]:
            if self.offer_to_worker(worker, task_id):
                accepted_by, dist_km = worker, d
                break
            else:
                yield self.env.timeout(1)
        if not accepted_by:
            self.log({"time": self.env.now, "event": "task_unaccepted", "task_id": task_id, "property": prop.name, "type": task_type_name})
            return
        travel_mins = travel_minutes_km(dist_km, accepted_by.speed_kmph)
        yield self.env.timeout(travel_mins)
        work_duration = self.sample_duration(ttype.mean_duration)
        yield self.env.timeout(work_duration)
        completed = self.complete_task(accepted_by, task_id)
        finish_time = self.env.now
        wait_minutes = finish_time - start_time
        payout = self.pay(ttype.base_payout, wait_minutes, ttype.sla_minutes) if completed else 0.0
        self.log({
            "time": finish_time, "event": "task_completed" if completed else "task_failed",
            "task_id": task_id, "property": prop.name, "type": task_type_name,
            "worker_id": accepted_by.id, "distance_km": round(dist_km,2),
            "travel_minutes": round(travel_mins,1), "work_minutes": round(work_duration,1),
            "wait_minutes": round(wait_minutes,1), "within_sla": wait_minutes <= ttype.sla_minutes,
            "payout": round(payout,2)
        })

def build_default_scenario(num_properties=25, num_workers=120):
    task_types = {
        "Maintenance": TaskType("Maintenance", 60, "tech", 35.0, 240),
        "Cleaning": TaskType("Cleaning", 45, "clean", 25.0, 180),
        "Audit": TaskType("Audit", 20, "audit", 12.0, 120),
        "Marketing": TaskType("Marketing", 30, "promo", 18.0, 180),
    }
    center_lat, center_lon = 40.7128, -74.0060
    properties, workers = [], []
    for i in range(num_properties):
        lat = center_lat + random.uniform(-0.3, 0.3)
        lon = center_lon + random.uniform(-0.3, 0.3)
        rates = {"Maintenance":0.2,"Cleaning":0.3,"Audit":0.4,"Marketing":0.1}
        properties.append(PropertyNode(i, f"Property-{i:03d}", lat, lon, rates))
    possible_skills = [["tech"],["clean"],["audit"],["promo"],["tech","clean"],["clean","audit"],["tech","promo"]]
    for i in range(num_workers):
        lat = center_lat + random.uniform(-0.35,0.35)
        lon = center_lon + random.uniform(-0.35,0.35)
        skills = random.choice(possible_skills)
        speed = random.uniform(15,30)
        acc = random.uniform(0.8,0.98)
        rel = random.uniform(0.9,0.995)
        workers.append(Worker(i, f"Worker-{i:03d}", lat, lon, skills, speed, acc, rel))
    return properties, workers, task_types

def run_sim(sim_minutes=12*60):
    env = simpy.Environment()
    props, workers, tasks = build_default_scenario()
    sim = TaskOSSim(env, props, workers, tasks)
    env.run(until=sim_minutes)
    return pd.DataFrame(sim.event_log)
