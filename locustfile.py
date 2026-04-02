from locust import HttpUser, task, between
import os

class CheckUpUser(HttpUser):
    wait_time = between(1, 3)

    @task(1)
    def health(self):
        self.client.get("/health")

    @task(3)
    def predict(self):
        test_dir = os.path.join(os.path.dirname(__file__), 'data', 'test')
        for f in os.listdir(test_dir):
            if f.endswith('.wav'):
                wav_path = os.path.join(test_dir, f)
                with open(wav_path, 'rb') as wav:
                    self.client.post(
                        "/predict",
                        files={"file": (f, wav, "audio/wav")}
                    )
                break