# -*- coding: utf-8 -*-
"""PerfectDou 原版推理轻量 API.

保留官方仓库原版 .so 与 onnx 推理链路，只暴露 perfectdou agent。
不加载 torch / rlcard，也不提供 douzero / rlcard / random 分支。
"""
import os
import random
import re
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from perfectdou.env.game import GameEnv, EnvCard2RealCard
from perfectdou.evaluation.perfectdou_agent import PerfectDouAgent

REPO = os.environ.get("PD_REPO", "/app/PerfectDou")
POSITIONS = ("landlord", "landlord_up", "landlord_down")

_agents = {}


def get_agent(position: str) -> PerfectDouAgent:
    if position not in POSITIONS:
        raise HTTPException(status_code=400,
                            detail="position 必须是 landlord / landlord_up / landlord_down")
    if position not in _agents:
        _agents[position] = PerfectDouAgent(position)
    return _agents[position]


def new_deck() -> List[int]:
    deck = []
    for i in range(3, 15):
        deck.extend([i] * 4)
    deck.extend([17] * 4)
    deck.extend([20, 30])
    return deck


def random_deal() -> Dict[str, List[int]]:
    deck = new_deck()
    random.shuffle(deck)
    return {
        "landlord": sorted(deck[:20]),
        "landlord_up": sorted(deck[20:37]),
        "landlord_down": sorted(deck[37:54]),
        "three_landlord_cards": sorted(deck[17:20]),
    }


def build_infoset(req: "PlayRequest") -> GameEnv:
    if req.position not in POSITIONS:
        raise HTTPException(status_code=400,
                            detail="position 必须是 landlord / landlord_up / landlord_down")
    for position in POSITIONS:
        if position not in req.hands:
            raise HTTPException(status_code=400, detail="hands 缺少 " + position)

    env = GameEnv({})
    for position in POSITIONS:
        env.info_sets[position].player_hand_cards = sorted(req.hands[position])
    env.three_landlord_cards = list(req.three_landlord_cards)
    env.bomb_num = req.bomb_num
    env.last_pid = req.last_pid or "landlord"
    env.played_cards = {p: [] for p in POSITIONS}
    env.last_move_dict = {p: [] for p in POSITIONS}
    env.card_play_action_seq = []

    for item in req.history:
        if item.position not in POSITIONS:
            raise HTTPException(status_code=400, detail="history 中出现非法 position")
        cards = sorted(item.cards)
        if cards:
            env.played_cards[item.position] += cards
            env.last_move_dict[item.position] = cards
        env.card_play_action_seq.append(cards)

    env.acting_player_position = req.position
    return env


class HistoryItem(BaseModel):
    position: str
    cards: List[int] = Field(default_factory=list)


class PlayRequest(BaseModel):
    position: str
    hands: Dict[str, List[int]]
    three_landlord_cards: List[int] = Field(default_factory=list)
    history: List[HistoryItem] = Field(default_factory=list)
    bomb_num: int = 0
    last_pid: Optional[str] = None


class PlayResponse(BaseModel):
    cards: List[int]
    real_cards: List[str]
    is_pass: bool


class GameRequest(BaseModel):
    seed: Optional[int] = None


class EvalRequest(BaseModel):
    num_games: int = 1000
    num_workers: int = 1
    timeout: int = 900


app = FastAPI(title="PerfectDou 原版推理 API")


@app.get("/health")
def health():
    return {"ok": True, "repo": REPO, "agent": "perfectdou"}


@app.post("/play", response_model=PlayResponse)
def play(req: PlayRequest):
    env = build_infoset(req)
    infoset = env.get_infoset()
    action = sorted(get_agent(req.position).act(infoset))
    return PlayResponse(
        cards=action,
        real_cards=[EnvCard2RealCard[c] for c in action],
        is_pass=len(action) == 0,
    )


@app.post("/game")
def game(req: GameRequest):
    if req.seed is not None:
        random.seed(req.seed)
    deal = random_deal()
    # 修正：card_play_init 直接引用 deal 里的列表，出牌会原地消耗它们。
    # 先保存一份初始发牌快照，返回时用它，避免返回的 deal 变成残局。
    initial_deal = {k: list(v) for k, v in deal.items()}
    players = {position: get_agent(position) for position in POSITIONS}
    env = GameEnv(players)
    env.card_play_init(deal)

    steps = 0
    while not env.game_over and steps < 2000:
        env.step()
        steps += 1

    return {
        "winner": env.get_winner() if env.game_over else None,
        "utility": env.player_utility_dict,
        "bomb_num": env.bomb_num,
        "steps": steps,
        "deal": {k: [EnvCard2RealCard[c] for c in v] for k, v in initial_deal.items()},
        "history": [[EnvCard2RealCard[c] for c in a] for a in env.card_play_action_seq],
    }


def parse_eval(text: str) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    pairs = re.findall(r"landlord\s*:\s*Farmers\s*-\s*(-?[\d.]+)\s*:\s*(-?[\d.]+)", text)
    return (pairs[0], pairs[1]) if len(pairs) >= 2 else (None, None)


@app.post("/evaluate")
def evaluate(req: EvalRequest):
    if req.num_workers < 1 or req.num_workers > 4:
        raise HTTPException(status_code=400, detail="num_workers 建议 1~4")
    if req.num_games < 1 or req.num_games > 100000:
        raise HTTPException(status_code=400, detail="num_games 范围 1~100000")

    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = REPO + os.pathsep + child_env.get("PYTHONPATH", "")

    try:
        with tempfile.TemporaryDirectory(prefix="pd_eval_") as tmp:
            data_path = os.path.join(tmp, "eval_data")
            gen = subprocess.run(
                [sys.executable, os.path.join(REPO, "generate_eval_data.py"),
                 "--output", data_path, "--num_games", str(req.num_games)],
                cwd=REPO, env=child_env, capture_output=True, text=True, timeout=req.timeout,
            )
            if gen.returncode != 0:
                raise HTTPException(status_code=500, detail="生成评估数据失败: " + gen.stderr[-2000:])

            proc = subprocess.run(
                [sys.executable, os.path.join(REPO, "evaluate.py"),
                 "--landlord", "perfectdou",
                 "--landlord_up", "perfectdou",
                 "--landlord_down", "perfectdou",
                 "--eval_data", data_path + ".pkl",
                 "--num_workers", str(req.num_workers)],
                cwd=REPO, env=child_env, capture_output=True, text=True, timeout=req.timeout,
            )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="评估超时，请减小 num_games")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail="评估失败: " + proc.stderr[-2000:])

    wp, adp = parse_eval(proc.stdout)
    return {
        "wp": {"landlord": wp[0] if wp else None, "farmers": wp[1] if wp else None},
        "adp": {"landlord": adp[0] if adp else None, "farmers": adp[1] if adp else None},
        "stdout": proc.stdout[-4000:],
    }
