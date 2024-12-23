import uiautomator2 as u2

class Zepp:
    def __init__(self, mobile_sn: str, input_resource_id: str, send_resource_id: str):
        self.d = u2.connect(mobile_sn)
        self.input_ele = self.d(resourceId=input_resource_id)
        self.send_ele = self.d(resourceId=send_resource_id)

    def send(self, cmd: str):
        self.input_ele.set_text(cmd)
        self.send_ele.click()
        print(f"send cmd: {cmd}")