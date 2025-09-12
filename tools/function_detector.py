#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
函数检测脚本
检测enhanced_multiturn_converter.py中定义的函数是否在JSON数据文件中出现过
"""

import json
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

class FunctionDetector:
    def __init__(self, json_file_path: str):
        """
        初始化函数检测器
        
        Args:
            json_file_path: JSON数据文件路径
        """
        self.json_file_path = json_file_path
        self.function_mapping = {
            # FileSystem相关
            'ls': 'file_system-ls',
            'pwd': 'file_system-pwd', 
            'cd': 'file_system-cd',
            'mkdir': 'file_system-mkdir',
            'touch': 'file_system-touch',
            'echo': 'file_system-echo',
            'cat': 'file_system-cat',
            'rm': 'file_system-rm',
            'mv': 'file_system-mv',
            'cp': 'file_system-cp',
            'find': 'file_system-find',
            'grep': 'file_system-grep',
            'tail': 'file_system-tail',
            'diff': 'file_system-diff',
            'wc': 'file_system-wc',
            'sort': 'file_system-sort',
            'du': 'file_system-du',
            'rmdir': 'file_system-rmdir',
            'load_scenario': 'file_system-load_scenario',
            'save_scenario': 'file_system-save_scenario',
            
            # Math相关
            'logarithm': 'math-logarithm',
            'mean': 'math-mean',
            'standard_deviation': 'math-standard_deviation',
            'si_unit_conversion': 'math-si_unit_conversion',
            'imperial_si_conversion': 'math-imperial_si_conversion',
            'add': 'math-add',
            'subtract': 'math-subtract',
            'multiply': 'math-multiply',
            'divide': 'math-divide',
            'power': 'math-power',
            'square_root': 'math-square_root',
            'absolute_value': 'math-absolute_value',
            'round_number': 'math-round_number',
            'percentage': 'math-percentage',
            'min_value': 'math-min_value',
            'max_value': 'math-max_value',
            'sum_values': 'math-sum_values',
            
            # posting/Posting相关
            'authenticate_twitter': 'posting-authenticate_twitter',
            'posting_get_login_status': 'posting-posting_get_login_status',
            'post_tweet': 'posting-post_tweet',
            'retweet': 'posting-retweet',
            'comment': 'posting-comment',
            'mention': 'posting-mention',
            'follow_user': 'posting-follow_user',
            'list_all_following': 'posting-list_all_following',
            'unfollow_user': 'posting-unfollow_user',
            'get_tweet': 'posting-get_tweet',
            'get_user_tweets': 'posting-get_user_tweets',
            'search_tweets': 'posting-search_tweets',
            'get_tweet_comments': 'posting-get_tweet_comments',
            'get_user_stats': 'posting-get_user_stats',
            
            # Ticket相关
            'create_ticket': 'ticket-create_ticket',
            'get_ticket': 'ticket-get_ticket',
            'close_ticket': 'ticket-close_ticket',
            'resolve_ticket': 'ticket-resolve_ticket',
            'edit_ticket': 'ticket-edit_ticket',
            'ticket_login': 'ticket-ticket_login',
            'ticket_get_login_status': 'ticket-ticket_get_login_status',
            'logout': 'ticket-logout',
            'get_user_tickets': 'ticket-get_user_tickets',
            
            # Trading相关
            'get_current_time': 'trading-get_current_time',
            'update_market_status': 'trading-update_market_status',
            'get_symbol_by_name': 'trading-get_symbol_by_name',
            'get_stock_info': 'trading-get_stock_info',
            'get_order_details': 'trading-get_order_details',
            'cancel_order': 'trading-cancel_order',
            'place_order': 'trading-place_order',
            'make_transaction': 'trading-make_transaction',
            'get_account_info': 'trading-get_account_info',
            'trading_login': 'trading-trading_login',
            'trading_get_login_status': 'trading-trading_get_login_status',
            'trading_logout': 'trading-trading_logout',
            'fund_account': 'trading-fund_account',
            'remove_stock_from_watchlist': 'trading-remove_stock_from_watchlist',
            'get_watchlist': 'trading-get_watchlist',
            'get_order_history': 'trading-get_order_history',
            'get_transaction_history': 'trading-get_transaction_history',
            'update_stock_price': 'trading-update_stock_price',
            'get_available_stocks': 'trading-get_available_stocks',
            'filter_stocks_by_price': 'trading-filter_stocks_by_price',
            'add_to_watchlist': 'trading-add_to_watchlist',
            'notify_price_change': 'trading-notify_price_change',
            
            # Travel相关
            'authenticate_travel': 'travel-authenticate_travel',
            'travel_get_login_status': 'travel-travel_get_login_status',
            'get_budget_fiscal_year': 'travel-get_budget_fiscal_year',
            'register_credit_card': 'travel-register_credit_card',
            'get_flight_cost': 'travel-get_flight_cost',
            'get_credit_card_balance': 'travel-get_credit_card_balance',
            'book_flight': 'travel-book_flight',
            'retrieve_invoice': 'travel-retrieve_invoice',
            'list_all_airports': 'travel-list_all_airports',
            'cancel_booking': 'travel-cancel_booking',
            'compute_exchange_rate': 'travel-compute_exchange_rate',
            'verify_traveler_information': 'travel-verify_traveler_information',
            'set_budget_limit': 'travel-set_budget_limit',
            'get_nearest_airport_by_city': 'travel-get_nearest_airport_by_city',
            'purchase_insurance': 'travel-purchase_insurance',
            'contact_customer_support': 'travel-contact_customer_support',
            'get_all_credit_cards': 'travel-get_all_credit_cards',
            
            # Vehicle相关
            'startEngine': 'vehicle-startEngine',
            'fillFuelTank': 'vehicle-fillFuelTank',
            'lockDoors': 'vehicle-lockDoors',
            'adjustClimateControl': 'vehicle-adjustClimateControl',
            'get_outside_temperature_from_google': 'vehicle-get_outside_temperature_from_google',
            'get_outside_temperature_from_weather_com': 'vehicle-get_outside_temperature_from_weather_com',
            'setHeadlights': 'vehicle-setHeadlights',
            'displayCarStatus': 'vehicle-displayCarStatus',
            'activateParkingBrake': 'vehicle-activateParkingBrake',
            'pressBrakePedal': 'vehicle-pressBrakePedal',
            'releaseBrakePedal': 'vehicle-releaseBrakePedal',
            'setCruiseControl': 'vehicle-setCruiseControl',
            'get_current_speed': 'vehicle-get_current_speed',
            'estimate_drive_feasibility_by_mileage': 'vehicle-estimate_drive_feasibility_by_mileage',
            'liter_to_gallon': 'vehicle-liter_to_gallon',
            'gallon_to_liter': 'vehicle-gallon_to_liter',
            'estimate_distance': 'vehicle-estimate_distance',
            'get_zipcode_based_on_city': 'vehicle-get_zipcode_based_on_city',
            'set_navigation': 'vehicle-set_navigation',
            'check_tire_pressure': 'vehicle-check_tire_pressure',
            'find_nearest_tire_shop': 'vehicle-find_nearest_tire_shop',
            
            # Message相关
            'list_users': 'message-list_users',
            'get_user_id': 'message-get_user_id',
            'message_login': 'message-message_login',
            'message_get_login_status': 'message-message_get_login_status',
            'send_message': 'message-send_message',
            'delete_message': 'message-delete_message',
            'view_messages_sent': 'message-view_messages_sent',
            'add_contact': 'message-add_contact',
            'search_messages': 'message-search_messages',
            'get_message_stats': 'message-get_message_stats',
        }
        
        # 统计结果
        self.found_functions = set()  # 找到的函数
        self.not_found_functions = set()  # 未找到的函数
        self.function_occurrences = defaultdict(int)  # 函数出现次数
        self.function_locations = defaultdict(list)  # 函数出现位置
        
    def load_json_data(self) -> List[Dict]:
        """加载JSON数据"""
        print(f"正在加载JSON文件: {self.json_file_path}")
        try:
            data = []
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # 跳过空行
                        try:
                            record = json.loads(line)
                            data.append(record)
                        except json.JSONDecodeError as e:
                            print(f"第 {line_num} 行JSON解析失败: {e}")
                            continue
            
            print(f"成功加载JSON文件，包含 {len(data)} 条记录")
            return data
        except Exception as e:
            print(f"加载JSON文件失败: {e}")
            return []
    
    def extract_functions_from_text(self, text: str) -> Set[str]:
        """
        从文本中提取函数调用
        
        Args:
            text: 要搜索的文本
            
        Returns:
            找到的函数集合
        """
        functions = set()
        
        # 匹配函数调用模式: function_name(parameters)
        pattern = r'([a-zA-Z_][a-zA-Z0-9_-]*(?:-[a-zA-Z0-9_-]+)*)\s*\('
        matches = re.findall(pattern, text)
        
        for match in matches:
            functions.add(match)
            
        return functions
    
    def search_functions_in_data(self, data: List[Dict]) -> None:
        """
        在数据中搜索函数
        
        Args:
            data: JSON数据列表
        """
        print("开始搜索函数...")
        
        # 获取所有新格式函数名
        new_format_functions = set(self.function_mapping.values())
        
        for idx, record in enumerate(data):
            # 将记录转换为字符串进行搜索
            record_text = json.dumps(record, ensure_ascii=False)
            
            # 提取文本中的函数
            found_functions = self.extract_functions_from_text(record_text)
            
            # 检查是否包含我们定义的函数
            for func in found_functions:
                if func in new_format_functions:
                    self.found_functions.add(func)
                    self.function_occurrences[func] += 1
                    # 确保record是字典类型
                    if isinstance(record, dict):
                        record_id = record.get('id', f'record_{idx}')
                    else:
                        record_id = f'record_{idx}'
                    self.function_locations[func].append({
                        'record_id': record_id,
                        'record_index': idx
                    })
        
        # 计算未找到的函数
        self.not_found_functions = new_format_functions - self.found_functions
        
        print(f"搜索完成！")
        print(f"找到 {len(self.found_functions)} 个函数")
        print(f"未找到 {len(self.not_found_functions)} 个函数")
    
    def generate_report(self) -> str:
        """生成检测报告"""
        report = []
        report.append("=" * 80)
        report.append("函数检测报告")
        report.append("=" * 80)
        report.append(f"JSON文件: {self.json_file_path}")
        report.append(f"检测时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 总体统计
        total_functions = len(self.function_mapping)
        found_count = len(self.found_functions)
        not_found_count = len(self.not_found_functions)
        
        report.append("总体统计:")
        report.append(f"  总函数数: {total_functions}")
        report.append(f"  找到的函数: {found_count} ({found_count/total_functions*100:.1f}%)")
        report.append(f"  未找到的函数: {not_found_count} ({not_found_count/total_functions*100:.1f}%)")
        report.append("")
        
        # 按类别统计
        categories = {
            'file_system': [],
            'math': [],
            'posting': [],
            'ticket': [],
            'trading': [],
            'travel': [],
            'vehicle': [],
            'message': []
        }
        
        for func in self.found_functions:
            category = func.split('-')[0]
            if category in categories:
                categories[category].append(func)
        
        report.append("按类别统计找到的函数:")
        for category, funcs in categories.items():
            if funcs:
                report.append(f"  {category}: {len(funcs)} 个函数")
                for func in sorted(funcs):
                    count = self.function_occurrences[func]
                    report.append(f"    - {func} (出现 {count} 次)")
        report.append("")
        
        # 详细统计
        report.append("详细统计:")
        report.append("-" * 40)
        
        # 找到的函数
        if self.found_functions:
            report.append("找到的函数:")
            for func in sorted(self.found_functions):
                count = self.function_occurrences[func]
                report.append(f"  ✓ {func} (出现 {count} 次)")
                # 显示前3个出现位置
                locations = self.function_locations[func][:3]
                for loc in locations:
                    report.append(f"    - 位置: {loc['record_id']} (记录索引: {loc['record_index']})")
                if len(self.function_locations[func]) > 3:
                    report.append(f"    - ... 还有 {len(self.function_locations[func]) - 3} 个位置")
        else:
            report.append("找到的函数: 无")
        
        report.append("")
        
        # 未找到的函数
        if self.not_found_functions:
            report.append("未找到的函数:")
            for func in sorted(self.not_found_functions):
                report.append(f"  ✗ {func}")
        else:
            report.append("未找到的函数: 无")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def run_detection(self) -> None:
        """运行完整的检测流程"""
        print("开始函数检测...")
        
        # 加载数据
        data = self.load_json_data()
        if not data:
            print("无法加载数据，检测终止")
            return
        
        # 搜索函数
        self.search_functions_in_data(data)
        
        # 生成报告
        report = self.generate_report()
        
        # 输出报告
        print("\n" + report)
        
        # 保存报告到文件
        report_file = "/home/ma-user/work/RL-Factory/function_detection_report.txt"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n报告已保存到: {report_file}")
        except Exception as e:
            print(f"保存报告失败: {e}")

def main():
    """主函数"""
    json_file_path = "/home/ma-user/work/RL-Factory/data/BFCL/multi-turn/enhanced_single_turn_data_with_execution_train170_test30.json"
    
    detector = FunctionDetector(json_file_path)
    detector.run_detection()

if __name__ == "__main__":
    main()
