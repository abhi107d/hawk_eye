import subprocess

def run_command(command):
    """Run a system command."""
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        exit(1)



def install_cuda_toolkit():
    """Install CUDA Toolkit and cuDNN."""
    print("Installing CUDA Toolkit and cuDNN...")
    run_command("conda install -c conda-forge cudatoolkit=11.8 cudnn=8.9.7 -y")

def install_pytorch():
    """Install PyTorch with GPU support."""
    print("Installing PyTorch with GPU support...")
    run_command("conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y")

def install_additional_dependencies():
    """Install additional dependencies."""
    print("Installing additional dependencies...")
    run_command("pip install -r requirements.txt")

def main():
    """Main setup process."""
    install_cuda_toolkit()
    install_pytorch()
    install_additional_dependencies()
    print("\nSetup completed successfully!")

if __name__ == "__main__":
    main()
