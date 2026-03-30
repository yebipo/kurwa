import os
import sys
import ast
import warnings
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error


# --- 1. ЖЕСТКИЕ ПАТЧИ ---
class Mock:
    def __init__(self, *args, **kwargs): pass

    def __getattr__(self, _): return Mock()

    def __call__(self, *args, **kwargs): return np.zeros((1, 1))


sys.modules['shap'] = Mock()
sys.modules['shap.explainers'] = Mock()

try:
    import torch

    _original_load = torch.load


    def safe_cpu_load(f, *args, **kwargs):
        kwargs['map_location'] = 'cpu'
        kwargs['weights_only'] = False
        return _original_load(f, *args, **kwargs)


    torch.load = safe_cpu_load
except ImportError:
    pass

warnings.filterwarnings('ignore')


# --- 2. УМНЫЙ АВТО-ХИЛЕР (ЛЕНИВАЯ ЗАГРУЗКА ЗАВИСИМОСТЕЙ) ---
def get_node_source(file_path, target_name):
    """Ищет конкретный класс, функцию или импорт по всем файлам, не трогая остальной код"""
    try:
        tree = ast.parse(file_path.read_text(encoding='utf-8'))
        for node in tree.body:
            # Если это определение класса или функции
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                if node.name == target_name: return ast.unparse(node)
            # Если это переменная/константа
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == target_name:
                        return ast.unparse(node)
            # Если это импорт (from module import Class)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if (alias.asname or alias.name) == target_name:
                        return ast.unparse(node)
    except Exception:
        pass
    return None


def auto_heal_load_and_predict(model_path, py_files, X_df):
    """Пытается загрузить и предиктить. При ошибках точечно подтягивает нужный кусок кода."""
    model = None
    py_files_sorted = sorted(py_files, key=os.path.getmtime, reverse=True)

    # ЭТАП 1: ЗАГРУЗКА И ВНЕДРЕНИЕ КЛАССОВ (до 15 попыток)
    for _ in range(15):
        try:
            model = joblib.load(model_path)
            break
        except AttributeError as e:
            error_msg = str(e)
            if "Can't get attribute" in error_msg:
                try:
                    missing_name = error_msg.split("'")[1]
                except IndexError:
                    print(f"  ❌ Ошибка: {e}")
                    return None

                print(f"  ⚕️ Лечим загрузку: Ищу '{missing_name}'...")
                found = False
                for py_file in py_files_sorted:
                    source = get_node_source(py_file, missing_name)
                    if source:
                        print(f"    -> Найдено в {py_file.name}. Внедряю!")
                        exec(source, globals())
                        found = True
                        break
                if not found:
                    print(f"    ❌ '{missing_name}' не найден ни в одном скрипте!")
                    return None
            else:
                print(f"  ❌ Критическая ошибка joblib: {e}")
                return None
        except Exception as e:
            print(f"  ❌ Ошибка загрузки: {e}")
            return None

    if model is None: return None

    # ЭТАП 2: ПРЕДИКТ (С подстраховками форматов)
    x_32 = np.asarray(np.nan_to_num(X_df.values, nan=0.0), dtype=np.float32)

    for _ in range(15):
        try:
            # 0. PyTorch-модели
            if 'torch' in sys.modules and isinstance(model, torch.nn.Module):
                model.eval()
                with torch.no_grad():
                    res = model(torch.tensor(x_32))
                    return res.numpy().flatten()

            # 1. Пробуем Pandas DataFrame (спасает от ошибки "only integers...")
            try:
                res = model.predict(X_df)
                if isinstance(res, tuple): res = res[0]
                if hasattr(res, 'ndim') and res.ndim > 1: res = res[:, 0]
                return res
            except NameError as e:
                raise e  # Кидаем хилеру
            except Exception:
                pass

            # 2. Пробуем Numpy массивы
            try:
                res = model.predict(x_32)
                if isinstance(res, tuple): res = res[0]
                if hasattr(res, 'ndim') and res.ndim > 1: res = res[:, 0]
                return res
            except NameError as e:
                raise e  # Кидаем хилеру
            except Exception:
                # 3. Решейп для 1D моделей
                return model.predict(x_32[:, 0].reshape(-1, 1))

        except NameError as e:
            # Если внутри predict() вызывается неизвестная функция
            missing_name = str(e).split("'")[1]
            print(f"  ⚕️ Лечим предикт: Ищу '{missing_name}'...")
            found = False
            for py_file in py_files_sorted:
                source = get_node_source(py_file, missing_name)
                if source:
                    print(f"    -> Найдено в {py_file.name}. Внедряю!")
                    exec(source, globals())
                    found = True
                    break
            if not found:
                print(f"    ❌ '{missing_name}' не найден ни в одном скрипте!")
                return None
        except Exception as e:
            print(f"  [Predict Error] {e}")
            return None

    return None


# --- 3. ЗАПУСК ---
def run_final_audit():
    output_dir = Path('audit_results')
    plots_dir = output_dir / 'plots'
    output_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    print("📂 Загрузка данных...")
    df = pd.read_csv('million.csv', sep=';')
    X = df.iloc[:, 1:5]
    y_true = np.asarray(df.iloc[:, -1].values, dtype=np.float32)

    results = []
    current_file = Path(__file__).name if '__file__' in globals() else 'doc.py'
    py_files = [f for f in Path('.').glob('*.py') if f.name != current_file]

    model_files = list(Path('.').glob('*.joblib'))
    print(f"🚀 Старт! Найдено .py скриптов: {len(py_files)}")

    base_globals = set(globals().keys())

    for model_path in model_files:
        m_name = model_path.stem
        print(f"\nМодель: {m_name}")

        # Полная очистка памяти перед загрузкой новой модели (защита от отравления имён)
        current_globals = set(globals().keys())
        for key in current_globals - base_globals:
            del globals()[key]

        try:
            y_pred = auto_heal_load_and_predict(model_path, py_files, X)

            if y_pred is None:
                continue

            y_pred = np.asarray(y_pred, dtype=np.float32).flatten()
            y_pred = np.clip(np.nan_to_num(y_pred, nan=0.5, posinf=5.0, neginf=-5.0), -5, 5)

            rmse_full = np.sqrt(mean_squared_error(y_true, y_pred))
            r2_full = r2_score(y_true, y_pred)
            bias_full = np.mean(y_pred - y_true)

            mask = (y_true >= 0.8) & (y_true <= 1.0)
            if mask.any():
                rmse_slice = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
                bias_slice = np.mean(y_pred[mask] - y_true[mask])
            else:
                rmse_slice, bias_slice = 0, 0

            results.append({
                'Model': m_name,
                'RMSE_0.8_1.0': round(rmse_slice, 6),
                'Bias_0.8_1.0': round(bias_slice, 6),
                'RMSE_Full': round(rmse_full, 6),
                'Bias_Full': round(bias_full, 6),
                'R2_Full': round(r2_full, 4)
            })

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            sns.kdeplot(y_true, label='True', fill=True, color='gray')
            sns.kdeplot(y_pred, label='Pred', fill=True, color='orange')
            plt.title(f'Bias: {bias_slice:.5f}\nRMSE[0.8-1]: {rmse_slice:.4f}')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.scatter(y_true[::500], y_pred[::500], alpha=0.3, s=2)
            plt.plot([0, 1], [0, 1], 'r--')
            plt.title(f'Scatter: {m_name}')

            plt.tight_layout()
            plt.savefig(plots_dir / f"{m_name}.png", dpi=100)
            plt.close()

            print(f"✅ Успех | RMSE 0.8-1: {rmse_slice:.5f} | Bias: {bias_slice:.5f}")

        except Exception as e:
            print(f"❌ Ошибка аудита: {str(e)[:100]}")

    if results:
        res_df = pd.DataFrame(results).sort_values('RMSE_0.8_1.0')
        res_df.to_csv(output_dir / 'AUDIT_FINAL_V15.csv', index=False, sep=';', encoding='utf-16')
        print("\n🏆 ТОП-10 МОДЕЛЕЙ (RMSE 0.8-1):")
        print(res_df[['Model', 'RMSE_0.8_1.0', 'Bias_0.8_1.0']].head(10))


if __name__ == "__main__":
    run_final_audit()