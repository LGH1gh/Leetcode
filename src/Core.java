class Person {
    int type = 1;
}

class Teacher extends Person {
    int type = 2;
}

class Student extends Person {
    int type = 3;
}

public class Core {
    public static void main(String[] args) {
        Person[] a = new Person[3];
        a[0] = new Teacher();
        a[1] = new Teacher();
        a[2] = new Teacher();

        Class cl = a.getClass();
        System.out.println(cl.getComponentType());
    }
}
