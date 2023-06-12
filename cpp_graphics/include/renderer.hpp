class CGLRenderer {
  public:
    CGLRenderer(int width, int height);
    ~CGLRenderer();

    virtual int*** render() = 0;

    int getWidth();
    int getHeight();

  private:
    int width, height;
    
  protected:
    int** pixels;

};
