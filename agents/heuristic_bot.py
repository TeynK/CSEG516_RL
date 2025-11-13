# agents/heuristic_bot.py

from typing import Optional, Dict, List, Tuple
from collections import defaultdict

from splendor_game.board import Board
from splendor_game.player import Player
from splendor_game.actions import Action, ActionType, get_legal_actions, get_legal_return_gems_actions
from splendor_game.card import DevelopmentCard, GemColor, CostDict
from splendor_game.constants import WINNING_SCORE

class HeuristicBot:
    def __init__(self, player_id: int):
        self.player_id: int = player_id
        self.target_card: Optional[DevelopmentCard] = None
        self.name = f"HeuristicBot (Player {player_id})"

    def choose_action(self, board: Board, player: Player, legal_actions: List[Action]) -> Optional[Action]:
        if not legal_actions:
            return None
            
        # 0b. 보석 반납 상태 처리
        if legal_actions[0].action_type == ActionType.RETURN_GEMS:
            # print(f"[{self.name}] 0b순위: 보석을 반납합니다.")
            return legal_actions[0]

        # --- 1순위: 즉시 승리 ---
        action = self._find_winning_move(board, player)
        if action:
            # print(f"[{self.name}] 1순위: 승리 수를 찾았습니다!")
            return action

        # --- 2순위: 예약된 '타겟' 구매 ---
        # (P3에서 L3, L2, L1 중 하나를 타겟으로 설정했을 수 있음)
        action = self._buy_reserved_target(player)
        if action:
            return action
        action = self._set_and_reserve_target(board, player)
        if action:
            return action
        best_buy_action = None
        lowest_cost = 99
        
        all_buyable_cards_with_actions = self._get_all_buyable_cards(board, player)
        
        for card, action in all_buyable_cards_with_actions:
            # P2에서 타겟 구매를 실패했으므로, 타겟이 아닌 카드만 구매
            if card != self.target_card:
                cost = sum(card.cost.values())
                if cost < lowest_cost:
                    lowest_cost = cost
                    best_buy_action = action
                        
        if best_buy_action:
            # print(f"[{self.name}] 5a순위: 가장 저렴한 '엔진' 카드를 구매합니다.")
            return best_buy_action
            
        # 5b. (살 수 있는 카드가 없을 때) 보드 위 L1 카드 예약 (골드 획득)
        if player.can_reserve():
            best_reserve_action = None
            lowest_cost = 99
            
            for i, card in enumerate(board.face_up_cards[1]): # L1 카드만
                if card:
                    cost = sum(card.cost.values())
                    if cost < lowest_cost:
                        lowest_cost = cost
                        best_reserve_action = Action(
                            action_type=ActionType.RESERVE_CARD,
                            card=card,
                            level=1,
                            index=i,
                            is_deck_reserve=False
                        )
            
            if best_reserve_action:
                # print(f"[{self.name}] 5b순위: L1 카드를 예약합니다. (골드 획득)")
                return best_reserve_action

        # 5c. 3개 보석 가져오기
        for action in legal_actions:
            if action.action_type == ActionType.TAKE_THREE_GEMS:
                # print(f"[{self.name}] 5c순위: 3개 보석을 가져옵니다.")
                return action

        # 5d. 2개 보석 가져오기
        for action in legal_actions:
            if action.action_type == ActionType.TAKE_TWO_GEMS:
                # print(f"[{self.name}] 5d순위: 2개 보석을 가져옵니다.")
                return action
                
        # 5e. (최후의 최후) 덱에서 예약
        for action in legal_actions:
            if action.action_type == ActionType.RESERVE_CARD and action.is_deck_reserve:
                 # print(f"[{self.name}] 5e순위: 덱에서 예약을 합니다.")
                return action

        # 5f. 위 모든 것이 실패하면
        # print(f"[{self.name}] 5f순위: 마지막 남은 행동을 실행합니다.")
        return legal_actions[0] if legal_actions else None

    def _get_all_buyable_cards(self, board: Board, player: Player) -> List[Tuple[DevelopmentCard, Action]]:
        """구매 가능한 모든 카드(보드, 예약)와 그에 맞는 Action을 리스트로 반환합니다."""
        buyable = []
        
        # 1. 예약된 카드 확인
        for i, card in enumerate(player.reserved_cards):
            if player.can_afford(card):
                action = Action(ActionType.BUY_CARD, card=card, index=i, is_reserved_buy=True)
                buyable.append((card, action))
                
        # 2. 보드 위의 카드 확인
        for level in board.face_up_cards:
            for i, card in enumerate(board.face_up_cards[level]):
                if card and player.can_afford(card):
                    action = Action(ActionType.BUY_CARD, card=card, level=level, index=i, is_reserved_buy=False)
                    buyable.append((card, action))
        return buyable

    def _find_winning_move(self, board: Board, player: Player) -> Optional[Action]:
        """1순위 로직: 15점을 넘기는 구매 액션을 찾습니다."""
        all_buyable = self._get_all_buyable_cards(board, player)
        for card, action in all_buyable:
            if (player.score + card.points) >= WINNING_SCORE:
                return action
        return None

    def _buy_reserved_target(self, player: Player) -> Optional[Action]:
        """2순위 로직: 예약된 타겟 카드를 구매합니다."""
        if not self.target_card:
            return None # 타겟이 없음

        for i, card in enumerate(player.reserved_cards):
            if card == self.target_card: # 이 카드가 내 타겟인가?
                if player.can_afford(card):
                    self.target_card = None # 타겟 달성, 초기화
                    return Action(ActionType.BUY_CARD, card=card, index=i, is_reserved_buy=True)
                else:
                    return None # 타겟이 있지만 아직 살 수 없음
        
        # 타겟 카드가 예약 목록에 없으면 (오류 또는 타겟 변경), 타겟 초기화
        self.target_card = None 
        return None

    def _calculate_cost_score(self, card: DevelopmentCard) -> float:
        """카드의 '가성비'를 계산합니다. (점수 / 비용)"""
        cost_sum = sum(card.cost.values())
        # 비용이 0인 카드는 없지만, 0으로 나누는 것을 방지
        return card.points / (cost_sum + 0.1) 

    def _set_and_reserve_target(self, board: Board, player: Player) -> Optional[Action]:
        """
        3순위 로직: 타겟이 없으면 카드를 예약합니다.
        [수정] L3 -> L2 -> L1 순서로 반드시 하나의 타겟을 찾습니다.
        """
        if self.target_card or not player.can_reserve():
            return None # 이미 타겟이 있거나, 예약을 못함

        best_card: Optional[DevelopmentCard] = None
        best_index: int = -1
        best_level: int = -1
        best_score: float = -1.0 # (가성비 점수)

        # --- 1. L3 카드 시도 (cost < 10) ---
        for i, card in enumerate(board.face_up_cards[3]):
            if card:
                effective_cost_dict = player.calculate_effective_cost(card)
                total_effective_cost = sum(effective_cost_dict.values())
                
                if total_effective_cost < 10: # [L3 필터]
                    score = self._calculate_cost_score(card)
                    if score > best_score:
                        best_score = score
                        best_card = card
                        best_index = i
                        best_level = 3
        
        # --- 2. L3 타겟을 못 찾았다면, L2 카드로 전략 변경 (cost < 8) ---
        if best_card is None:
            # print(f"[{self.name}] P3: L3 타겟 없음. L2 러시로 전략 변경.")
            best_score = -1.0 # 점수 초기화
            for i, card in enumerate(board.face_up_cards[2]): # 2단계 카드
                if card:
                    effective_cost_dict = player.calculate_effective_cost(card)
                    total_effective_cost = sum(effective_cost_dict.values())
                    
                    if total_effective_cost < 8: # [L2 필터]
                        score = self._calculate_cost_score(card) # 가성비
                        if score > best_score:
                            best_score = score
                            best_card = card
                            best_index = i
                            best_level = 2

        # --- 3. [수정] L2 타겟도 못 찾았다면, L1 카드로 전략 변경 ---
        if best_card is None:
            # print(f"[{self.name}] P3: L2 타겟 없음. L1 엔진 빌딩으로 전략 변경.")
            best_score = -1.0 # 점수 초기화
            lowest_cost = 99 # L1은 가성비 대신 '가장 싼' 카드를 타겟
            
            for i, card in enumerate(board.face_up_cards[1]): # 1단계 카드
                if card:
                    # L1은 필터 없음 (무조건 하나 잡음)
                    cost = sum(card.cost.values())
                    if cost < lowest_cost:
                        lowest_cost = cost
                        best_card = card
                        best_index = i
                        best_level = 1

        # --- 4. 최종 타겟 설정 ---
        if best_card:
            self.target_card = best_card # L3, L2, 또는 L1 카드를 타겟으로 설정!
            return Action(ActionType.RESERVE_CARD, card=best_card, level=best_level, index=best_index, is_deck_reserve=False)
        
        # (만약 L1 카드조차 4장 모두 비어있는 극단적 상황이라면, P5로 넘어감)
        return None
    
    def _get_needed_gems(self, player: Player) -> CostDict:
        """현재 타겟을 사기 위해 '순수하게' 부족한 보석을 계산합니다."""
        needed_gems = defaultdict(int)
        if not self.target_card:
            return needed_gems

        effective_cost = player.calculate_effective_cost(self.target_card)
        
        for color, cost in effective_cost.items():
            current_gems = player.gems.get(color, 0)
            if current_gems < cost:
                needed_gems[color] = cost - current_gems
                
        return needed_gems

    def _gather_for_target(self, board: Board, player: Player, legal_actions: List[Action]) -> Optional[Action]:
        """4순위 로직: 설정된 타겟을 위해 자원을 모읍니다."""
        if not self.target_card:
            return None

        needed_gems = self._get_needed_gems(player)
        if not needed_gems:
            # (이론상 2순위 로직에서 걸러졌어야 하지만, 안전장치)
            return self._buy_reserved_target(player)

        # 4a. '저격': 2개 보석 가져오기 (가장 부족한 보석)
        sorted_needed = sorted(needed_gems.items(), key=lambda item: item[1], reverse=True)
        most_needed_color, _ = sorted_needed[0]

        if board.gem_stacks.get(most_needed_color, 0) >= 4:
            # 2개 가져오기가 가능한지 확인 (legal_actions를 순회할 필요 없음)
            return Action(ActionType.TAKE_TWO_GEMS, gems={most_needed_color: 2})

        # 4b. '징검다리': 필요한 보너스를 주는 L1 카드 구매
        needed_bonus_color = most_needed_color
        for i, card in enumerate(board.face_up_cards[1]): # 1단계 카드만 봅니다
            if card and card.gem_type == needed_bonus_color and player.can_afford(card):
                return Action(ActionType.BUY_CARD, card=card, level=1, index=i, is_reserved_buy=False)

        # 4c. '자원 확보': 필요한 보석 3개 가져오기
        colors_to_take = []
        for color, _ in sorted_needed: # 필요한 보석 목록
            if board.gem_stacks.get(color, 0) > 0:
                colors_to_take.append(color)
                
        if len(colors_to_take) >= 3:
            gem_dict = {colors_to_take[0]: 1, colors_to_take[1]: 1, colors_to_take[2]: 1}
            return Action(ActionType.TAKE_THREE_GEMS, gems=gem_dict)
        elif 0 < len(colors_to_take) < 3:
            gem_dict = {c: 1 for c in colors_to_take}
            available_gems = GemColor.get_standard_gems()
            for color in available_gems:
                if color not in gem_dict and board.gem_stacks.get(color, 0) > 0:
                    gem_dict[color] = 1
                    if len(gem_dict) == 3:
                        break
            
            # 3개가 안 되더라도(보석 고갈) 1개라도 가져올 수 있다면 시도
            if len(gem_dict) >= 1: 
                 return Action(ActionType.TAKE_THREE_GEMS, gems=gem_dict)
        
        # 4순위에서 할 수 있는 게 없으면 None을 반환하고 5순위로 넘어갑니다.
        return None