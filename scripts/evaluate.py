import traceback
from splendor_game.game import SplendorGame
from splendor_game.player import Player
from splendor_game.actions import get_legal_return_gems_actions, ActionType, Action
from agents.baselines.heuristic_bot import HeuristicBot

MAX_GAME_TURNS = 5000

def run_bots_game(num_players=2, verbose=True):
    try:
        game = SplendorGame(num_players=num_players)
    except ValueError as e:
        print(f"게임 생성 오류: {e}")
        return
        
    bots = [HeuristicBot(player_id=i) for i in range(num_players)]
    
    if verbose:
        print(f"--- {num_players}인 봇 대전 시작 ---")
        print(f"봇 0: {bots[0].name}")
        print(f"봇 1: {bots[1].name}")
        print("="*30)
        
    turn_count = 0

    while not game.game_over:
        
        if turn_count > MAX_GAME_TURNS:
            if verbose:
                print("\n" + "="*30)
                print(f"!!!! 무한 반복 감지 !!!!")
                print(f"게임이 {MAX_GAME_TURNS}턴을 초과했습니다. 게임을 강제 종료합니다.")
                print("="*30)
            return None

        current_player: Player = game.get_current_player()
        bot: HeuristicBot = bots[current_player.player_id]

        if verbose:
            print(f"\n--- 턴 {turn_count // num_players + 1} - {bot.name} ---")
            print(f"현재 점수: {current_player.score}")
            print(f"봇 타겟 카드: {bot.target_card}")

        try:
            if game.current_player_state == "RETURN_GEMS":
                legal_actions = get_legal_return_gems_actions(current_player)
                if verbose: print("상태: 보석 반납")
            else:
                legal_actions = game.get_legal_actions()
                
            if not legal_actions:
                print("\n" + "="*50)
                print(f"!!!! 치명적인 버그 감지: 유효한 행동 없음 !!!!")
                print(f"턴: {turn_count}, 플레이어: {current_player.player_id}")
                print("--- 플레이어 상태 ---")
                print(current_player)
                print(f"예약 카드 개수: {len(current_player.reserved_cards)}")
                print("--- 보드 상태 ---")
                print(game.board)
                print("="*50)
                raise Exception("유효한 행동이 없는 상태(deadlock)에 도달했습니다.")
                continue

            selected_action = bot.choose_action(game.board, current_player, legal_actions)

            if selected_action is None:
                if verbose: print(f"[{bot.name}] 봇이 None을 반환. 안전한 Fallback 실행...")
                 
                safe_action_found = False
                for action in legal_actions:
                    if action.action_type in [ActionType.TAKE_THREE_GEMS, ActionType.TAKE_TWO_GEMS, ActionType.RETURN_GEMS] or \
                       (action.action_type == ActionType.RESERVE_CARD and action.is_deck_reserve):
                        selected_action = action
                        safe_action_found = True
                        break
                 
                if not safe_action_found:
                    action_template = legal_actions[0]
                    
                    if action_template.action_type == ActionType.RESERVE_CARD:
                        card_to_reserve = game.board.face_up_cards[action_template.level][action_template.index]
                        if card_to_reserve:
                            selected_action = Action(action_type=ActionType.RESERVE_CARD, card=card_to_reserve, level=action_template.level, index=action_template.index, is_deck_reserve=False)
                        else:
                            selected_action = action_template

                    elif action_template.action_type == ActionType.BUY_CARD:
                        card_to_buy = None
                        if action_template.is_reserved_buy:
                            if action_template.index < len(current_player.reserved_cards):
                                card_to_buy = current_player.reserved_cards[action_template.index]
                        else:
                            card_to_buy = game.board.face_up_cards[action_template.level][action_template.index]
                        
                        if card_to_buy:
                             selected_action = Action(action_type=ActionType.BUY_CARD, card=card_to_buy, level=action_template.level, index=action_template.index, is_reserved_buy=action_template.is_reserved_buy)
                        else:
                            selected_action = action_template
                    else:
                        selected_action = action_template
            
            if verbose:
                print(f"선택한 행동: {selected_action}")

            game.step(selected_action)
            turn_count += 1

        except Exception as e:
            print(f"\n!!!! 치명적인 오류 발생 !!!!")
            print(f"{bot.name}이(가) 행동 결정 중 오류 발생: {e}") 
            print(f"현재 플레이어 상태: {current_player}")
            print(f"현재 보드 상태: {game.board}")
            traceback.print_exc()
            return None

    if verbose:
        print("\n" + "="*30)
        print("--- 게임 종료 ---")
        print(f"총 턴 수: {turn_count}")
        print(f"승자: {bots[game.winner_id].name if game.winner_id is not None else '무승부'}")
        print("="*30)
        
    return game.winner_id

if __name__ == "__main__":
    
    num_games = 1000
    winners = {0: 0, 1: 0, None: 0}
    
    print("="*40)
    print(f"휴리스틱 봇 대전 테스트 ({num_games} 게임)")
    print("="*40)
    
    for i in range(num_games):
        print(f"\n--- [ 게임 {i+1} / {num_games} ] ---")
        
        is_verbose = (i == 0) 
        if not is_verbose:
            print("... (verbose=False로 실행 중) ...")
        
        winner_id = run_bots_game(num_players=2, verbose=is_verbose) 
        
        winners[winner_id] += 1
        
        if not is_verbose:
            print(f"게임 {i+1} 종료. 승자: {winner_id if winner_id is not None else '오류/무승부'}")

        
    print("\n\n--- 최종 결과 요약 ---")
    print(f"총 {num_games} 게임 실행")
    print(f"봇 0 (선공) 승리: {winners[0]}회")
    print(f"봇 1 (후공) 승리: {winners[1]}회")
    print(f"오류/무승부: {winners[None]}회")