class Beverage:
    def __init__(self, name, price):
        self.name = name
        self.price = price

    def calculate(self, quantity):
        total_price = self.price * quantity
        return total_price


menu = {
    "커피": Beverage("커피", 3000),
    "녹차": Beverage("녹차", 2500),
    "아이스티": Beverage("아이스티", 3000)
}


order = {}
while True:
    drink = input("주문할 음료를 선택하세요 (종료하려면 '종료' 입력): ")
    if drink == "종료":
        break

    if drink in menu:
        quantity = int(input(f"{drink}의 수량을 입력하세요: "))
        order[drink] = quantity
    else:
        print("메뉴에 없는 음료입니다. 다시 선택해주세요.")


total_order_price = 0
for drink, quantity in order.items():
    beverage = menu[drink]
    price = beverage.calculate(quantity)
    total_order_price += price
    print(f"{drink} {quantity}잔의 가격은 {price}원입니다.")

print(f"총 주문 가격은 {total_order_price}원입니다.")